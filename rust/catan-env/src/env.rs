//! The RL environment: codec + observation encoder + reward plumbing around
//! one `CatanGame`, and a rayon-parallel vector of them.
//!
//! Semantics (turn-based multi-agent, PettingZoo-AEC shaped):
//! - `step(action_id)` executes the acting seat's action and advances to the
//!   NEXT seat that must decide (turn owner, trade responder, or discarder).
//!   The returned reward is what that next seat accrued since ITS last
//!   decision — the trainer pairs it with that seat's previous action.
//! - Forced moves (exactly one legal action: must-roll, lone steal-skip,
//!   forced discards) are resolved internally when `auto_resolve_forced` is
//!   on: the policy never sees a non-decision, and dice remain internal
//!   chance nodes. Property: every observation has >= 2 legal actions.
//! - On termination every seat's outstanding reward (shaped accruals +
//!   terminal win/loss) is delivered in one `terminal_rewards` vector.
//! - Truncation at `max_turns` ends the episode with no winner and no
//!   terminal bonus (shaped accruals still flush).

use std::sync::Arc;

use catan_core::eval::income_by_resource;
use catan_core::game::{Action, CatanGame};
use catan_core::players::{HeuristicPlayer, Player, RandomPlayer, RolloutBot};
use catan_core::replay::GameRecord;
use catan_core::state::NUM_RESOURCES;

use crate::alpha::{AlphaBot, AlphaConfig};
use crate::codec::{decode_action, encode_action, NUM_ACTIONS};
use crate::net::MlpNet;
use crate::obs::{encode_obs, Visibility, OBS_DIM};

#[derive(Debug, Clone, Copy)]
pub struct RewardConfig {
    /// Terminal reward for the winner.
    pub win: f32,
    /// Terminal reward for every other seat.
    pub loss: f32,
    /// Shaped reward per victory point gained (0.0 = pure terminal reward).
    /// Bootstrap crutch for early training; anneal toward 0.
    pub vp_delta: f32,
    /// Potential-based shaping scale. Rewards are issued as
    /// `scale * (Phi(next) - Phi(current))` and the potential is forced to
    /// zero at episode end, so complete-episode shaping return telescopes to
    /// zero instead of changing the terminal objective.
    pub potential_scale: f32,
    /// Split the configured loss across the losing seats. With win=1 and
    /// loss=-1, four-player terminal rewards become +1/-1/3 and sum to zero.
    pub zero_sum: bool,
}

impl Default for RewardConfig {
    fn default() -> Self {
        RewardConfig {
            win: 1.0,
            loss: -1.0,
            vp_delta: 0.0,
            potential_scale: 0.0,
            zero_sum: false,
        }
    }
}

/// Simple progress estimate used only through potential differences.
///
/// Victory points dominate. The remaining weight rewards expected resource
/// acquisition and whether that production supports at least one coherent
/// route to more VP:
/// - expansion: road + settlement resources,
/// - city: wheat + stone,
/// - development cards: wheat + sheep + stone.
///
/// Current hand contents, ports, road length, army progress, and opponent
/// strength are deliberately excluded.
pub fn opinionated_potential(game: &CatanGame, player: usize) -> f32 {
    if game.is_game_over() {
        return 0.0;
    }
    let state = &game.state;
    simple_potential(
        state.calculate_victory_points(player),
        state.victory_target,
        income_by_resource(state, player),
    )
}

fn simple_potential(vp: i32, victory_target: i32, income: [f32; NUM_RESOURCES]) -> f32 {
    // Income indexes: wheat, sheep, wood, brick, stone.
    let [wheat, sheep, wood, brick, stone] = income;

    // A strong position produces roughly 1.5 cards per roll. Clamp so raw
    // production remains a secondary signal after VP.
    let acquisition = (income.iter().sum::<f32>() / 1.5).clamp(0.0, 1.0);

    // Expected complete purchases per roll for three broad strategies.
    let expansion = (wood / 2.0).min(brick / 2.0).min(wheat).min(sheep);
    let city = (wheat / 2.0).min(stone / 3.0);
    let development = wheat.min(sheep).min(stone);
    let strategy = (expansion.max(city).max(development) / 0.10).clamp(0.0, 1.0);

    let vp_progress = (vp as f32 / victory_target.max(1) as f32).clamp(0.0, 1.0);
    0.90 * vp_progress + 0.06 * acquisition + 0.04 * strategy
}

#[cfg(test)]
mod potential_tests {
    use super::simple_potential;

    #[test]
    fn victory_points_dominate_resource_terms() {
        let no_income = [0.0; 5];
        let max_income = [1.0; 5];
        assert!(simple_potential(1, 7, no_income) > simple_potential(0, 7, max_income));
    }

    #[test]
    fn coherent_strategy_beats_unusable_income_at_equal_total() {
        let city_engine = [0.30, 0.0, 0.0, 0.0, 0.30];
        let disconnected = [0.0, 0.30, 0.30, 0.0, 0.0];
        assert!(simple_potential(2, 7, city_engine) > simple_potential(2, 7, disconnected));
    }

    #[test]
    fn potential_is_bounded() {
        assert_eq!(simple_potential(10, 10, [10.0; 5]), 1.0);
        assert_eq!(simple_potential(0, 10, [0.0; 5]), 0.0);
    }
}

/// Who controls a seat. Bot seats are auto-played inside `advance()` at
/// engine speed — the policy only ever sees decision points for `Policy`
/// seats. Used for evaluation (policy vs fixed opponents) and
/// mixed-opponent training.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeatKind {
    Policy,
    HeuristicBot,
    RandomBot,
    /// The GA-evolved heuristic (frozen Elo anchor).
    HeuristicV2Bot,
    /// Flat Monte Carlo search (strong but slow — eval anchor, not a
    /// training opponent at scale).
    RolloutBot,
    /// The shipped policy-prior-guided search agent. Evaluation-only.
    AlphaBot,
}

#[derive(Clone)]
pub struct EnvConfig {
    pub num_players: usize,
    pub victory_target: i32,
    pub visibility: Visibility,
    pub reward: RewardConfig,
    pub auto_resolve_forced: bool,
    pub max_turns: u32,
    pub seat_kinds: [SeatKind; 4],
    /// Auto-play policy-controlled initial settlements with Heuristic-v2.
    /// Setup roads and every later decision remain policy-controlled.
    pub policy_opening_heuristic: bool,
    pub alpha_net: Option<Arc<MlpNet>>,
    pub alpha: AlphaConfig,
}

impl Default for EnvConfig {
    fn default() -> Self {
        EnvConfig {
            num_players: 4,
            victory_target: 10,
            visibility: Visibility::Perfect,
            reward: RewardConfig::default(),
            auto_resolve_forced: true,
            max_turns: 1000,
            seat_kinds: [SeatKind::Policy; 4],
            policy_opening_heuristic: false,
            alpha_net: None,
            alpha: AlphaConfig::default(),
        }
    }
}

#[derive(Clone)]
enum Bot {
    Heuristic(HeuristicPlayer),
    Random(RandomPlayer),
    Rollout(Box<RolloutBot>),
    Alpha(Box<AlphaBot>),
}

impl Bot {
    fn choose(&mut self, game: &CatanGame, valid: &[Action]) -> Action {
        match self {
            Bot::Heuristic(p) => p.choose_action(game, valid),
            Bot::Random(p) => p.choose_action(game, valid),
            Bot::Rollout(p) => p.choose_action(game, valid),
            Bot::Alpha(p) => p.choose_action(game, valid),
        }
    }
}

/// What `step`/`reset` hand back. Observation and mask are written via
/// `write_obs`/`write_mask` into caller buffers (zero-copy for the batch
/// path).
#[derive(Debug, Clone, Copy)]
pub struct StepResult {
    /// Seat that must act next (meaningless when `done`).
    pub seat: usize,
    /// Reward accrued by `seat` since its previous decision.
    pub reward: f32,
    pub done: bool,
    /// Winning seat, or -1 (truncation / not finished).
    pub winner: i8,
    /// Per-seat outstanding rewards, delivered once, when `done`.
    pub terminal_rewards: [f32; 4],
}

#[derive(Clone)]
pub struct CatanEnv {
    game: CatanGame,
    episode_seed: u64,
    pub config: EnvConfig,
    pending: [f32; 4],
    last_vp: [i32; 4],
    last_potential: [f32; 4],
    done: bool,
    scratch: Vec<Action>,
    recording: Option<GameRecord>,
    bots: [Option<Bot>; 4],
    policy_opening_bots: [Option<HeuristicPlayer>; 4],
    pub episodes: u64,
    pub steps: u64,
}

impl CatanEnv {
    pub fn new(config: EnvConfig, seed: u64) -> CatanEnv {
        assert!(
            config.seat_kinds[..config.num_players]
                .iter()
                .any(|k| *k == SeatKind::Policy),
            "at least one seat must be policy-controlled"
        );
        let mut env = CatanEnv {
            game: CatanGame::new_with_target(config.num_players, seed, config.victory_target),
            episode_seed: seed,
            config,
            pending: [0.0; 4],
            last_vp: [0; 4],
            last_potential: [0.0; 4],
            done: false,
            scratch: Vec::with_capacity(512),
            recording: None,
            bots: [None, None, None, None],
            policy_opening_bots: [None, None, None, None],
            episodes: 0,
            steps: 0,
        };
        env.reset(seed);
        env
    }

    /// Start a fresh episode. Returns the first decision point.
    pub fn reset(&mut self, seed: u64) -> StepResult {
        self.game =
            CatanGame::new_with_target(self.config.num_players, seed, self.config.victory_target);
        self.episode_seed = seed;
        self.game.record_history = false;
        self.pending = [0.0; 4];
        self.last_vp = [0; 4];
        self.last_potential = [0.0; 4];
        self.done = false;
        self.episodes += 1;
        self.initialize_bots(seed);
        if self.recording.is_some() {
            self.recording = Some(GameRecord::start(&self.game));
        }
        if self.config.reward.potential_scale != 0.0 {
            for p in 0..self.config.num_players {
                self.last_potential[p] = opinionated_potential(&self.game, p);
            }
        }
        self.advance()
    }

    pub fn episode_seed(&self) -> u64 {
        self.episode_seed
    }

    fn initialize_bots(&mut self, seed: u64) {
        // Independent bot RNG streams keep opponent sampling reproducible
        // without exposing the live episode's private RNG state to search.
        for s in 0..self.config.num_players {
            let bot_seed = seed ^ (s as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15);
            self.policy_opening_bots[s] = (self.config.policy_opening_heuristic
                && self.config.seat_kinds[s] == SeatKind::Policy)
                .then(|| HeuristicPlayer::v2(bot_seed ^ 0x0FE1_1A6));
            self.bots[s] = match self.config.seat_kinds[s] {
                SeatKind::Policy => None,
                SeatKind::HeuristicBot => Some(Bot::Heuristic(HeuristicPlayer::new(bot_seed))),
                SeatKind::RandomBot => Some(Bot::Random(RandomPlayer::new(bot_seed))),
                SeatKind::HeuristicV2Bot => Some(Bot::Heuristic(HeuristicPlayer::v2(bot_seed))),
                SeatKind::RolloutBot => {
                    Some(Bot::Rollout(Box::new(RolloutBot::new(bot_seed, 12, 40))))
                }
                SeatKind::AlphaBot => Some(Bot::Alpha(Box::new(AlphaBot::new_with_visibility(
                    bot_seed,
                    self.config
                        .alpha_net
                        .as_ref()
                        .expect("alpha seat requires alpha_net")
                        .clone(),
                    self.config.alpha.root_k,
                    self.config.alpha.samples,
                    self.config.alpha.depth,
                    self.config.visibility,
                )))),
            };
        }
    }

    /// Clone a live environment into one fair search sample.
    ///
    /// The observer's private cards and all public state are preserved.
    /// Opponent hidden cards, future chance, and bot RNG streams are sampled
    /// independently so search cannot inspect the real episode's secrets.
    pub fn search_sample(&self, observer: usize, seed: u64) -> CatanEnv {
        self.search_sample_with_alpha(observer, seed, self.config.alpha)
    }

    /// Build a fair search sample with a separately configured AlphaBot
    /// budget. Evaluation can keep the shipped opponent at full strength
    /// while internal search uses a cheaper policy-matched surrogate.
    pub fn search_sample_with_alpha(
        &self,
        observer: usize,
        seed: u64,
        alpha: AlphaConfig,
    ) -> CatanEnv {
        assert!(
            observer < self.config.num_players,
            "observer seat out of range"
        );
        let mut sample = self.clone();
        sample.recording = None;
        if sample.config.visibility == Visibility::Realistic {
            crate::search::redeterminize(&mut sample.game, observer, seed);
        }
        sample.config.alpha = alpha;
        sample.initialize_bots(seed ^ 0xB07A_6EED_5EED_1234);
        sample
    }

    /// Record this and future episodes as CTRP replays (video/eval source).
    pub fn enable_recording(&mut self) {
        self.recording = Some(GameRecord::start(&self.game));
    }

    /// Take the (finished or in-progress) record of the current episode.
    pub fn take_record(&mut self) -> Option<GameRecord> {
        let mut record = self.recording.take()?;
        record.finish(&self.game);
        Some(record)
    }

    pub fn game(&self) -> &CatanGame {
        &self.game
    }

    pub fn current_seat(&self) -> usize {
        self.game.current_player()
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Execute one policy decision. The id MUST be legal (the trainer
    /// samples under the mask); an illegal id is a masking bug and panics.
    pub fn step(&mut self, action_id: usize) -> StepResult {
        assert!(!self.done, "step() on a finished episode — reset first");
        let action = decode_action(&self.game, action_id);
        self.execute(&action);
        self.steps += 1;
        self.advance()
    }

    fn execute(&mut self, action: &Action) {
        let ok = match self.recording.as_mut() {
            Some(record) => record.record_step(&mut self.game, action),
            None => self.game.execute_action(action),
        };
        assert!(
            ok,
            "engine rejected {action:?} — the policy must sample under the legal mask"
        );
        if self.config.reward.vp_delta != 0.0 {
            self.accrue_vp_deltas();
        }
        if self.config.reward.potential_scale != 0.0 {
            self.accrue_potential_deltas();
        }
    }

    fn accrue_vp_deltas(&mut self) {
        for p in 0..self.config.num_players {
            let vp = self.game.state.calculate_victory_points(p);
            self.pending[p] += self.config.reward.vp_delta * (vp - self.last_vp[p]) as f32;
            self.last_vp[p] = vp;
        }
    }

    fn accrue_potential_deltas(&mut self) {
        for p in 0..self.config.num_players {
            let next = opinionated_potential(&self.game, p);
            self.pending[p] += self.config.reward.potential_scale * (next - self.last_potential[p]);
            self.last_potential[p] = next;
        }
    }

    /// Resolve forced moves and bot seats, detect termination, and surface
    /// the next POLICY decision point.
    fn advance(&mut self) -> StepResult {
        loop {
            if self.game.is_game_over() || self.game.state.turn >= self.config.max_turns {
                return self.finish();
            }
            let seat = self.game.current_player();
            // Bot seats play themselves at engine speed.
            if self.bots[seat].is_some() {
                self.game.fill_valid_actions(&mut self.scratch);
                debug_assert!(!self.scratch.is_empty(), "live phase with no actions");
                let action = match self.bots[seat].as_mut() {
                    Some(bot) => bot.choose(&self.game, &self.scratch),
                    None => unreachable!(),
                };
                let action = if action_executes(&self.game, &action) {
                    action
                } else {
                    self.scratch
                        .iter()
                        .copied()
                        .find(|candidate| action_executes(&self.game, candidate))
                        .expect("bot decision point must include an executable action")
                };
                self.execute(&action);
                continue;
            }
            if self.policy_opening_bots[seat].is_some() {
                self.game.fill_valid_actions(&mut self.scratch);
                let is_initial_settlement = !self.scratch.is_empty()
                    && self
                        .scratch
                        .iter()
                        .all(|action| matches!(action, Action::PlaceInitialSettlement { .. }));
                if is_initial_settlement {
                    let action = self.policy_opening_bots[seat]
                        .as_mut()
                        .expect("opening bot checked above")
                        .choose_action(&self.game, &self.scratch);
                    self.execute(&action);
                    continue;
                }
            }
            if !self.config.auto_resolve_forced {
                break;
            }
            self.game.fill_valid_actions(&mut self.scratch);
            if self.scratch.len() != 1 {
                break;
            }
            let forced = self.scratch[0];
            self.execute(&forced);
        }
        let seat = self.game.current_player();
        let reward = std::mem::take(&mut self.pending[seat]);
        StepResult {
            seat,
            reward,
            done: false,
            winner: -1,
            terminal_rewards: [0.0; 4],
        }
    }

    fn finish(&mut self) -> StepResult {
        self.done = true;
        let winner = self.game.winner();
        // Truncation is also a terminal state for the learning problem.
        // Close the potential so shaping sums to zero over the episode.
        if self.config.reward.potential_scale != 0.0 {
            for p in 0..self.config.num_players {
                self.pending[p] -= self.config.reward.potential_scale * self.last_potential[p];
                self.last_potential[p] = 0.0;
            }
        }
        if winner >= 0 {
            let loss = if self.config.reward.zero_sum && self.config.num_players > 1 {
                self.config.reward.loss / (self.config.num_players - 1) as f32
            } else {
                self.config.reward.loss
            };
            for p in 0..self.config.num_players {
                self.pending[p] += if p as i8 == winner {
                    self.config.reward.win
                } else {
                    loss
                };
            }
        }
        let terminal_rewards = std::mem::take(&mut self.pending);
        StepResult {
            seat: self.game.current_player(),
            reward: 0.0,
            done: true,
            winner,
            terminal_rewards,
        }
    }

    /// Encode the current decision point's observation for the acting seat.
    pub fn write_obs(&self, out: &mut [f32]) {
        encode_obs(&self.game, self.current_seat(), self.config.visibility, out);
    }

    /// Fill the legality mask for the current decision point. All-false when
    /// the episode is done.
    pub fn write_mask(&mut self, out: &mut [bool]) {
        assert_eq!(out.len(), NUM_ACTIONS);
        out.fill(false);
        if self.done {
            return;
        }
        self.game.fill_valid_actions(&mut self.scratch);
        for action in &self.scratch {
            out[encode_action(&self.game, action)] = true;
        }
    }
}

fn action_executes(game: &CatanGame, action: &Action) -> bool {
    let mut probe = game.clone();
    probe.execute_action(action)
}

// ---------------------------------------------------------------------------
// Vectorized environment
// ---------------------------------------------------------------------------

/// N independent environments stepped in one rayon-parallel call — the unit
/// a training loop talks to (one batched NN inference per `step_batch`).
/// Episodes auto-reset: when env i finishes, `dones[i]` is set, its
/// `terminal_rewards` row is filled, and the returned obs/mask/seat already
/// belong to the next episode's first decision.
pub struct VecCatanEnv {
    pub envs: Vec<CatanEnv>,
    terminal_road_targets: Vec<[f32; 5]>,
    episode_counters: Vec<u64>,
    base_seed: u64,
    /// Finished-episode CTRP records harvested during `step_batch` when
    /// recording is enabled (Mutex: pushed from the rayon loop, rare).
    completed_records: std::sync::Mutex<Vec<GameRecord>>,
    recording: bool,
    /// (turns, winner, final VPs, cap) per finished episode — dashboard fuel.
    episode_stats: std::sync::Mutex<Vec<(u32, i8, [u8; 4], bool)>>,
}

/// Deterministic per-episode seed stream (splitmix64).
fn episode_seed(base: u64, env_idx: u64, episode: u64) -> u64 {
    let mut z = base ^ (env_idx.wrapping_mul(0x9E3779B97F4A7C15)) ^ (episode << 32);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

impl VecCatanEnv {
    pub fn new(num_envs: usize, config: EnvConfig, base_seed: u64) -> VecCatanEnv {
        let envs = (0..num_envs)
            .map(|i| CatanEnv::new(config.clone(), episode_seed(base_seed, i as u64, 0)))
            .collect();
        VecCatanEnv {
            envs,
            terminal_road_targets: vec![[0.0; 5]; num_envs],
            episode_counters: vec![0; num_envs],
            base_seed,
            completed_records: std::sync::Mutex::new(Vec::new()),
            recording: false,
            episode_stats: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Take (turns, winner, final_vp, hit_turn_cap) for episodes finished
    /// since the last drain.
    pub fn drain_episode_stats(&mut self) -> Vec<(u32, i8, [u8; 4], bool)> {
        std::mem::take(&mut self.episode_stats.lock().unwrap())
    }

    /// Record every episode in every env as a CTRP replay; finished records
    /// are harvested via `drain_records`. Intended for evaluation runs (a
    /// few hundred games), not bulk training.
    pub fn enable_recording(&mut self) {
        self.recording = true;
        for env in &mut self.envs {
            env.enable_recording();
        }
    }

    /// Take all completed episode records harvested so far.
    pub fn drain_records(&mut self) -> Vec<GameRecord> {
        std::mem::take(&mut self.completed_records.lock().unwrap())
    }

    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// Absolute-seat terminal Longest Road targets for the most recent batch.
    ///
    /// Rows are one-hot over seats 0-3 plus class 4 for no holder. Rows are
    /// all zero for environments that did not terminate in the last step.
    pub fn terminal_road_targets(&self) -> &[[f32; 5]] {
        &self.terminal_road_targets
    }

    /// Write every env's current obs/mask/seat (after construction or to
    /// re-prime buffers).
    pub fn observe(&mut self, obs: &mut [f32], masks: &mut [bool], seats: &mut [u32]) {
        let n = self.envs.len();
        assert_eq!(obs.len(), n * OBS_DIM);
        assert_eq!(masks.len(), n * NUM_ACTIONS);
        assert_eq!(seats.len(), n);
        use rayon::prelude::*;
        self.envs
            .par_iter_mut()
            .zip(obs.par_chunks_mut(OBS_DIM))
            .zip(masks.par_chunks_mut(NUM_ACTIONS))
            .zip(seats.par_iter_mut())
            .for_each(|(((env, obs), mask), seat)| {
                env.write_obs(obs);
                env.write_mask(mask);
                *seat = env.current_seat() as u32;
            });
    }

    /// Step every env with its chosen action id. Buffer shapes:
    /// obs `N*OBS_DIM`, masks `N*NUM_ACTIONS`, seats/rewards/dones `N`,
    /// terminal_rewards `N*4` (rows valid only where `dones`).
    #[allow(clippy::too_many_arguments)]
    pub fn step_batch(
        &mut self,
        actions: &[u32],
        obs: &mut [f32],
        masks: &mut [bool],
        seats: &mut [u32],
        rewards: &mut [f32],
        dones: &mut [bool],
        terminal_rewards: &mut [f32],
    ) {
        let n = self.envs.len();
        assert_eq!(actions.len(), n);
        assert_eq!(obs.len(), n * OBS_DIM);
        assert_eq!(masks.len(), n * NUM_ACTIONS);
        assert_eq!(seats.len(), n);
        assert_eq!(rewards.len(), n);
        assert_eq!(dones.len(), n);
        assert_eq!(terminal_rewards.len(), n * 4);

        use rayon::prelude::*;
        let base_seed = self.base_seed;
        self.terminal_road_targets.fill([0.0; 5]);

        // One flat per-env lane so the parallel zip stays readable.
        struct Lane<'a> {
            env: &'a mut CatanEnv,
            episodes: &'a mut u64,
            action: u32,
            obs: &'a mut [f32],
            mask: &'a mut [bool],
            seat: &'a mut u32,
            reward: &'a mut f32,
            done: &'a mut bool,
            terminal: &'a mut [f32],
            road_target: &'a mut [f32; 5],
        }

        let lanes: Vec<Lane> = self
            .envs
            .iter_mut()
            .zip(self.episode_counters.iter_mut())
            .zip(actions.iter())
            .zip(obs.chunks_mut(OBS_DIM))
            .zip(masks.chunks_mut(NUM_ACTIONS))
            .zip(seats.iter_mut())
            .zip(rewards.iter_mut())
            .zip(dones.iter_mut())
            .zip(terminal_rewards.chunks_mut(4))
            .zip(self.terminal_road_targets.iter_mut())
            .map(
                |(
                    ((((((((env, episodes), &action), obs), mask), seat), reward), done), terminal),
                    road_target,
                )| {
                    Lane {
                        env,
                        episodes,
                        action,
                        obs,
                        mask,
                        seat,
                        reward,
                        done,
                        terminal,
                        road_target,
                    }
                },
            )
            .collect();

        let recording = self.recording;
        let completed = &self.completed_records;
        let stats = &self.episode_stats;
        lanes.into_par_iter().enumerate().for_each(|(i, lane)| {
            let result = lane.env.step(lane.action as usize);
            if result.done {
                *lane.done = true;
                *lane.reward = 0.0;
                lane.terminal.copy_from_slice(&result.terminal_rewards);
                let state = &lane.env.game().state;
                let road_class = if state.longest_road_player >= 0 {
                    state.longest_road_player as usize
                } else {
                    4
                };
                lane.road_target[road_class] = 1.0;
                let mut vps = [0u8; 4];
                for (p, vp) in vps.iter_mut().enumerate().take(state.num_players) {
                    *vp = state.calculate_victory_points(p) as u8;
                }
                stats.lock().unwrap().push((
                    state.turn,
                    result.winner,
                    vps,
                    state.turn >= lane.env.config.max_turns,
                ));
                if recording {
                    if let Some(record) = lane.env.take_record() {
                        completed.lock().unwrap().push(record);
                    }
                    lane.env.enable_recording();
                }
                *lane.episodes += 1;
                lane.env
                    .reset(episode_seed(base_seed, i as u64, *lane.episodes));
            } else {
                *lane.done = false;
                *lane.reward = result.reward;
                lane.terminal.fill(0.0);
            }
            lane.env.write_obs(lane.obs);
            lane.env.write_mask(lane.mask);
            *lane.seat = lane.env.current_seat() as u32;
        });
    }
}
