//! PyO3 boundary: the batched Catan RL environment for Python trainers.
//!
//! Design: ONE Python call per training step for the whole batch. The GIL
//! is released while Rust steps all envs in parallel; results come back as
//! fresh NumPy arrays (one memcpy per buffer — microseconds against an NN
//! forward pass). Checkpoints must store `codec_version`/`obs_version` and
//! refuse to run against mismatched layouts.

mod opponent_planner;

use catan_core::game::CatanGame;
use catan_core::players::Player;
use catan_env::alpha::AlphaBot;
use catan_env::env::SeatKind;
use catan_env::net::MlpNet;
use catan_env::{
    decode_action, encode_action, encode_obs, opinionated_potential, private_targets,
    redeterminize, CatanEnv, EnvConfig, RewardConfig, VecCatanEnv, Visibility,
    ACTION_TYPE_BOUNDARIES, CODEC_VERSION, NUM_ACTIONS, OBS_DIM, OBS_VERSION,
};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use opponent_planner::{OpeningHybridPolicy, OpponentAwarePlanner};

fn parse_seats(seats: Option<Vec<String>>, num_players: usize) -> PyResult<[SeatKind; 4]> {
    let mut kinds = [SeatKind::Policy; 4];
    if let Some(seats) = seats {
        if seats.len() != num_players {
            return Err(PyValueError::new_err(format!(
                "seats must list {num_players} entries"
            )));
        }
        for (i, s) in seats.iter().enumerate() {
            kinds[i] = match s.as_str() {
                "policy" => SeatKind::Policy,
                "heuristic" => SeatKind::HeuristicBot,
                "random" => SeatKind::RandomBot,
                "rollout" => SeatKind::RolloutBot,
                "heuristic_v2" => SeatKind::HeuristicV2Bot,
                "alpha" => SeatKind::AlphaBot,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "seat kind must be policy/heuristic/heuristic_v2/random/rollout/alpha, got '{other}'"
                    )))
                }
            };
        }
    }
    Ok(kinds)
}

fn parse_visibility(s: &str) -> PyResult<Visibility> {
    match s {
        "perfect" => Ok(Visibility::Perfect),
        "realistic" => Ok(Visibility::Realistic),
        other => Err(PyValueError::new_err(format!(
            "visibility must be 'perfect' or 'realistic', got '{other}'"
        ))),
    }
}

/// Batched vectorized Catan environment (see catan-env's VecCatanEnv).
#[pyclass]
pub(crate) struct VecEnv {
    pub(crate) inner: VecCatanEnv,
    obs: Vec<f32>,
    masks: Vec<bool>,
    seats: Vec<u32>,
    rewards: Vec<f32>,
    dones: Vec<bool>,
    terminals: Vec<f32>,
}

#[pyclass]
#[derive(Clone)]
struct SearchState {
    game: CatanGame,
    max_turns: u32,
}

/// AlphaBot search teacher over a SearchState. In realistic mode each call
/// samples one information set consistent with the acting player's knowledge.
#[pyclass]
struct AlphaTeacher {
    net: std::sync::Arc<MlpNet>,
    root_k: usize,
    samples: usize,
    depth: u32,
    visibility: Visibility,
}

#[pymethods]
impl AlphaTeacher {
    #[new]
    #[pyo3(signature = (net_path, root_k=8, samples=24, depth=80, visibility="realistic"))]
    fn new(
        net_path: String,
        root_k: usize,
        samples: usize,
        depth: u32,
        visibility: &str,
    ) -> PyResult<Self> {
        if root_k == 0 || samples == 0 {
            return Err(PyValueError::new_err("root_k and samples must be positive"));
        }
        Ok(AlphaTeacher {
            net: std::sync::Arc::new(MlpNet::load(net_path.as_ref())),
            root_k,
            samples,
            depth,
            visibility: parse_visibility(visibility)?,
        })
    }

    fn action(&self, py: Python<'_>, state: PyRef<'_, SearchState>, seed: u64) -> PyResult<usize> {
        if state.is_done() {
            return Err(PyValueError::new_err("teacher action on terminal state"));
        }
        let game = state.game.clone();
        let net = self.net.clone();
        let root_k = self.root_k;
        let samples = self.samples;
        let depth = self.depth;
        let visibility = self.visibility;
        py.allow_threads(move || {
            let mut valid = Vec::with_capacity(128);
            game.fill_valid_actions(&mut valid);
            let mut teacher =
                AlphaBot::new_with_visibility(seed, net, root_k, samples, depth, visibility);
            let action = teacher.choose_action(&game, &valid);
            Ok(encode_action(&game, &action))
        })
    }
}

impl SearchState {
    fn resolve_forced(&mut self) {
        let mut valid = Vec::with_capacity(64);
        while !self.game.is_game_over() && self.game.state.turn < self.max_turns {
            self.game.fill_valid_actions(&mut valid);
            if valid.len() != 1 {
                break;
            }
            let action = valid[0];
            assert!(self.game.execute_action(&action));
        }
    }
}

#[pymethods]
impl SearchState {
    fn copy(&self) -> SearchState {
        self.clone()
    }

    fn redeterminize(&mut self, observer: usize, seed: u64) -> PyResult<()> {
        if observer >= self.game.state.num_players {
            return Err(PyValueError::new_err("observer seat out of range"));
        }
        redeterminize(&mut self.game, observer, seed);
        Ok(())
    }

    fn obs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let mut buf = vec![0.0; OBS_DIM];
        encode_obs(
            &self.game,
            self.game.current_player(),
            Visibility::Realistic,
            &mut buf,
        );
        PyArray1::from_vec_bound(py, buf)
    }

    fn mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let mut valid = Vec::with_capacity(128);
        let mut mask = vec![false; NUM_ACTIONS];
        if !self.is_done() {
            self.game.fill_valid_actions(&mut valid);
            for action in &valid {
                mask[catan_env::encode_action(&self.game, action)] = true;
            }
        }
        PyArray1::from_vec_bound(py, mask)
    }

    /// Execute one encoded action and collapse subsequent forced moves.
    /// Returns (done, winner, next_seat).
    fn step(&mut self, action_id: usize) -> PyResult<(bool, i8, usize)> {
        if self.is_done() {
            return Err(PyValueError::new_err("step on terminal search state"));
        }
        if action_id >= NUM_ACTIONS {
            return Err(PyValueError::new_err("action id out of range"));
        }
        let action = decode_action(&self.game, action_id);
        if !self.game.execute_action(&action) {
            return Err(PyValueError::new_err("illegal action for search state"));
        }
        self.resolve_forced();
        Ok((
            self.is_done(),
            self.game.winner(),
            self.game.current_player(),
        ))
    }

    fn current_seat(&self) -> usize {
        self.game.current_player()
    }

    fn is_done(&self) -> bool {
        self.game.is_game_over() || self.game.state.turn >= self.max_turns
    }

    fn outcome(&self) -> [f32; 4] {
        let mut out = [0.0; 4];
        let winner = self.game.winner();
        if winner >= 0 {
            let loss = -1.0 / (self.game.state.num_players - 1) as f32;
            for value in out.iter_mut().take(self.game.state.num_players) {
                *value = loss;
            }
            out[winner as usize] = 1.0;
        }
        out
    }

    fn final_vp(&self) -> [f32; 4] {
        let mut out = [0.0; 4];
        for (p, value) in out.iter_mut().enumerate().take(self.game.state.num_players) {
            *value = self.game.state.calculate_victory_points(p) as f32
                / self.game.state.victory_target as f32;
        }
        out
    }

    /// Potential values in relative seat order: actor, +1, +2, +3.
    fn potential_values(&self) -> [f32; 4] {
        let mut out = [0.0; 4];
        let actor = self.game.current_player();
        for (rel, value) in out.iter_mut().enumerate().take(self.game.state.num_players) {
            *value = opinionated_potential(&self.game, (actor + rel) % self.game.state.num_players);
        }
        out
    }
}

impl VecEnv {
    fn arrays<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<bool>>,
        Bound<'py, PyArray1<u32>>,
    )> {
        let n = self.inner.num_envs();
        let obs = PyArray1::from_slice_bound(py, &self.obs).reshape([n, OBS_DIM])?;
        let masks = PyArray1::from_slice_bound(py, &self.masks).reshape([n, NUM_ACTIONS])?;
        let seats = PyArray1::from_slice_bound(py, &self.seats);
        Ok((obs, masks, seats))
    }
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (num_envs, num_players=4, victory_target=10, visibility="perfect",
                        vp_delta=0.0, potential_scale=0.0, win=1.0, loss=-1.0,
                        zero_sum=false, auto_resolve_forced=true,
                        max_turns=1000, seed=0, seats=None, alpha_net=None,
                        alpha_root_k=8, alpha_samples=96, alpha_depth=300,
                        policy_opening_heuristic=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_envs: usize,
        num_players: usize,
        victory_target: i32,
        visibility: &str,
        vp_delta: f32,
        potential_scale: f32,
        win: f32,
        loss: f32,
        zero_sum: bool,
        auto_resolve_forced: bool,
        max_turns: u32,
        seed: u64,
        seats: Option<Vec<String>>,
        alpha_net: Option<String>,
        alpha_root_k: usize,
        alpha_samples: usize,
        alpha_depth: u32,
        policy_opening_heuristic: bool,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be positive"));
        }
        if !(2..=4).contains(&num_players) {
            return Err(PyValueError::new_err("num_players must be 2-4"));
        }
        if !(3..=20).contains(&victory_target) {
            return Err(PyValueError::new_err("victory_target must be 3-20"));
        }
        if alpha_root_k == 0 || alpha_samples == 0 {
            return Err(PyValueError::new_err(
                "alpha_root_k and alpha_samples must be positive",
            ));
        }
        let config = EnvConfig {
            num_players,
            victory_target,
            visibility: parse_visibility(visibility)?,
            reward: RewardConfig {
                win,
                loss,
                vp_delta,
                potential_scale,
                zero_sum,
            },
            auto_resolve_forced,
            max_turns,
            seat_kinds: parse_seats(seats, num_players)?,
            policy_opening_heuristic,
            alpha_net: alpha_net
                .map(|path| std::sync::Arc::new(catan_env::net::MlpNet::load(path.as_ref()))),
            alpha: catan_env::AlphaConfig {
                root_k: alpha_root_k,
                samples: alpha_samples,
                depth: alpha_depth,
            },
        };
        Ok(VecEnv {
            inner: VecCatanEnv::new(num_envs, config, seed),
            obs: vec![0.0; num_envs * OBS_DIM],
            masks: vec![false; num_envs * NUM_ACTIONS],
            seats: vec![0; num_envs],
            rewards: vec![0.0; num_envs],
            dones: vec![false; num_envs],
            terminals: vec![0.0; num_envs * 4],
        })
    }

    /// Current (obs[N, OBS_DIM], masks[N, NUM_ACTIONS], seats[N]) without
    /// stepping — prime the first batch with this.
    fn observe<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<bool>>,
        Bound<'py, PyArray1<u32>>,
    )> {
        let (inner, obs, masks, seats) = (
            &mut self.inner,
            &mut self.obs,
            &mut self.masks,
            &mut self.seats,
        );
        py.allow_threads(|| inner.observe(obs, masks, seats));
        self.arrays(py)
    }

    /// Step every env. Returns (obs, masks, seats, rewards[N], dones[N],
    /// terminal_rewards[N, 4]); where dones[i], the obs/mask/seat already
    /// belong to env i's next auto-reset episode.
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'py, u32>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<bool>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray2<f32>>,
    )> {
        let n = self.inner.num_envs();
        let actions = actions.as_slice()?;
        if actions.len() != n {
            return Err(PyValueError::new_err(format!(
                "expected {n} actions, got {}",
                actions.len()
            )));
        }
        let actions: Vec<u32> = actions.to_vec();
        let (inner, obs, masks, seats, rewards, dones, terminals) = (
            &mut self.inner,
            &mut self.obs,
            &mut self.masks,
            &mut self.seats,
            &mut self.rewards,
            &mut self.dones,
            &mut self.terminals,
        );
        py.allow_threads(|| {
            inner.step_batch(&actions, obs, masks, seats, rewards, dones, terminals)
        });

        let (obs, masks, seats) = self.arrays(py)?;
        let rewards = PyArray1::from_slice_bound(py, &self.rewards);
        let dones = PyArray1::from_slice_bound(py, &self.dones);
        let terminals = PyArray1::from_slice_bound(py, &self.terminals).reshape([n, 4])?;
        Ok((obs, masks, seats, rewards, dones, terminals))
    }

    /// Record every episode as a CTRP replay (evaluation/video runs).
    fn enable_recording(&mut self) {
        self.inner.enable_recording();
    }

    /// (turns, winner, final_vp[4], hit_turn_cap) per episode finished
    /// since the last call — dashboard `game` events.
    fn take_episode_stats(&mut self) -> Vec<(u32, i8, [u8; 4], bool)> {
        self.inner.drain_episode_stats()
    }

    fn terminal_road_targets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let targets: Vec<f32> = self
            .inner
            .terminal_road_targets()
            .iter()
            .flatten()
            .copied()
            .collect();
        PyArray1::from_vec_bound(py, targets).reshape([self.inner.num_envs(), 5])
    }

    /// Harvest finished-episode CTRP replays as `bytes` objects.
    fn take_replays<'py>(&mut self, py: Python<'py>) -> Vec<Bound<'py, PyBytes>> {
        self.inner
            .drain_records()
            .into_iter()
            .map(|record| PyBytes::new_bound(py, &record.to_bytes()))
            .collect()
    }

    fn snapshot(&self, index: usize) -> PyResult<SearchState> {
        let env = self
            .inner
            .envs
            .get(index)
            .ok_or_else(|| PyValueError::new_err("environment index out of range"))?;
        Ok(SearchState {
            game: env.game().clone(),
            max_turns: env.config.max_turns,
        })
    }

    fn private_targets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let n = self.inner.num_envs();
        let mut targets = vec![0.0; n * 30];
        for (env, row) in self.inner.envs.iter().zip(targets.chunks_mut(30)) {
            private_targets(env.game(), env.current_seat(), row);
        }
        PyArray1::from_vec_bound(py, targets).reshape([n, 30])
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.num_envs()
    }
    #[getter]
    fn num_actions(&self) -> usize {
        NUM_ACTIONS
    }
    #[getter]
    fn obs_dim(&self) -> usize {
        OBS_DIM
    }
    #[getter]
    fn codec_version(&self) -> u32 {
        CODEC_VERSION
    }
    #[getter]
    fn obs_version(&self) -> u32 {
        OBS_VERSION
    }
}

/// Single-env handle for evaluation loops driving one game at a time.
#[pyclass]
pub(crate) struct Env {
    pub(crate) inner: CatanEnv,
}

#[pymethods]
impl Env {
    #[new]
    #[pyo3(signature = (num_players=4, victory_target=10, visibility="realistic",
                        potential_scale=0.0, zero_sum=true, max_turns=1000,
                        seed=0, seats=None, alpha_net=None,
                        alpha_root_k=8, alpha_samples=96, alpha_depth=300,
                        policy_opening_heuristic=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_players: usize,
        victory_target: i32,
        visibility: &str,
        potential_scale: f32,
        zero_sum: bool,
        max_turns: u32,
        seed: u64,
        seats: Option<Vec<String>>,
        alpha_net: Option<String>,
        alpha_root_k: usize,
        alpha_samples: usize,
        alpha_depth: u32,
        policy_opening_heuristic: bool,
    ) -> PyResult<Self> {
        if alpha_root_k == 0 || alpha_samples == 0 {
            return Err(PyValueError::new_err(
                "alpha_root_k and alpha_samples must be positive",
            ));
        }
        let config = EnvConfig {
            num_players,
            victory_target,
            visibility: parse_visibility(visibility)?,
            reward: RewardConfig {
                potential_scale,
                zero_sum,
                ..RewardConfig::default()
            },
            max_turns,
            seat_kinds: parse_seats(seats, num_players)?,
            policy_opening_heuristic,
            alpha_net: alpha_net
                .map(|path| std::sync::Arc::new(catan_env::net::MlpNet::load(path.as_ref()))),
            alpha: catan_env::AlphaConfig {
                root_k: alpha_root_k,
                samples: alpha_samples,
                depth: alpha_depth,
            },
            ..EnvConfig::default()
        };
        Ok(Env {
            inner: CatanEnv::new(config, seed),
        })
    }

    fn reset(&mut self, seed: u64) {
        self.inner.reset(seed);
    }

    /// Returns (seat, reward, done, winner, terminal_rewards[4]).
    fn step(&mut self, action: usize) -> (usize, f32, bool, i8, [f32; 4]) {
        let r = self.inner.step(action);
        (r.seat, r.reward, r.done, r.winner, r.terminal_rewards)
    }

    fn obs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let mut buf = vec![0.0f32; OBS_DIM];
        self.inner.write_obs(&mut buf);
        PyArray1::from_vec_bound(py, buf)
    }

    fn mask<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let mut buf = vec![false; NUM_ACTIONS];
        self.inner.write_mask(&mut buf);
        PyArray1::from_vec_bound(py, buf)
    }

    fn current_seat(&self) -> usize {
        self.inner.current_seat()
    }

    fn snapshot(&self) -> SearchState {
        SearchState {
            game: self.inner.game().clone(),
            max_turns: self.inner.config.max_turns,
        }
    }

    fn private_target<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let mut target = vec![0.0; 30];
        private_targets(self.inner.game(), self.inner.current_seat(), &mut target);
        PyArray1::from_vec_bound(py, target)
    }

    fn final_vp(&self) -> [f32; 4] {
        let state = &self.inner.game().state;
        let mut out = [0.0; 4];
        for (p, value) in out.iter_mut().enumerate().take(state.num_players) {
            *value = state.calculate_victory_points(p) as f32 / state.victory_target as f32;
        }
        out
    }
}

#[pymodule]
fn catan_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    m.add_class::<Env>()?;
    m.add_class::<SearchState>()?;
    m.add_class::<AlphaTeacher>()?;
    m.add_class::<OpponentAwarePlanner>()?;
    m.add_class::<OpeningHybridPolicy>()?;
    m.add("NUM_ACTIONS", NUM_ACTIONS)?;
    m.add("ACTION_TYPE_BOUNDARIES", ACTION_TYPE_BOUNDARIES.to_vec())?;
    m.add("OBS_DIM", OBS_DIM)?;
    m.add("CODEC_VERSION", CODEC_VERSION)?;
    m.add("OBS_VERSION", OBS_VERSION)?;
    Ok(())
}
