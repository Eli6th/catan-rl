//! Opponent-aware search over a cloned live environment.
//!
//! Unlike `AlphaTeacher`, this planner keeps the configured opponent
//! controllers in the simulation. Every rollout re-samples hidden cards,
//! chance, and opponent RNG, then lets `CatanEnv::step` advance through the
//! opponents until the candidate must act again.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::mem::Discriminant;
use std::sync::Arc;

use catan_core::board::topology;
use catan_core::eval::{evaluate_state, income_by_resource, ValueParams};
use catan_core::game::{Action, CatanGame, TurnPhase};
use catan_core::players::{HeuristicPlayer, Player};
use catan_core::state::{CITY_COST, DEV_CARD_COST, DEV_VICTORY_POINT, SETTLEMENT_COST};
use catan_env::env::SeatKind;
use catan_env::net::{MlpNet, NetScratch};
use catan_env::{decode_action, encode_action};
use catan_env::{opinionated_potential, AlphaConfig, CatanEnv, NUM_ACTIONS, OBS_DIM};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::{Env, VecEnv};

#[derive(Clone, Copy)]
struct PlannerConfig {
    root_k: usize,
    samples: usize,
    continuation_decisions: usize,
    value_weight: f32,
    potential_weight: f32,
    vp_gain_weight: f32,
    building_gain_weight: f32,
    road_control_weight: f32,
    common_random_numbers: bool,
    opponent_alpha: AlphaConfig,
}

#[derive(Clone, Copy, Default)]
struct RolloutMetrics {
    value: f32,
    vp_gain: f32,
    building_gain: f32,
    road_control: f32,
}

type ScoringState = (CatanEnv, usize, u64, usize);
type ActionMetricRow = (usize, f32, f32, f32, f32, f32);

impl RolloutMetrics {
    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            vp_gain: self.vp_gain + other.vp_gain,
            building_gain: self.building_gain + other.building_gain,
            road_control: self.road_control + other.road_control,
        }
    }

    fn scale(self, factor: f32) -> Self {
        Self {
            value: self.value * factor,
            vp_gain: self.vp_gain * factor,
            building_gain: self.building_gain * factor,
            road_control: self.road_control * factor,
        }
    }

    fn score(self, config: PlannerConfig) -> f32 {
        self.value
            + config.vp_gain_weight * self.vp_gain
            + config.building_gain_weight * self.building_gain
            + config.road_control_weight * self.road_control
    }
}

/// Search policy that models the actual configured opponents.
#[pyclass]
pub(crate) struct OpponentAwarePlanner {
    net: Arc<MlpNet>,
    config: PlannerConfig,
}

/// Neural policy with heuristic-only setup placement.
#[pyclass]
#[derive(Clone)]
pub(crate) struct OpeningHybridPolicy {
    net: Arc<MlpNet>,
    specialist_net: Option<Arc<MlpNet>>,
    specialist_net_min_vp: i32,
    specialist_net_seat_mask: u8,
    late_net: Option<Arc<MlpNet>>,
    late_net_min_vp: i32,
    late_net_seat_mask: u8,
    final_net: Option<Arc<MlpNet>>,
    final_net_min_vp: i32,
    final_net_min_cities: u8,
    final_net_max_cities: u8,
    final_net_seat_mask: u8,
    heuristic: HeuristicPlayer,
    settlement_neural_mix: f32,
    strategy_settlement_weight: f32,
    opening_production_weight: f32,
    opening_wheat_weight: f32,
    opening_wheat_seat_mask: u8,
    opening_city_weight: f32,
    opening_city_seat_mask: u8,
    opening_settlement_lookahead: bool,
    opening_rollout_candidates: usize,
    opening_rollout_samples: usize,
    opening_rollout_finalists: usize,
    opening_rollout_final_samples: usize,
    opening_rollout_prior_weight: f32,
    rollout_vp_margin_weight: f32,
    common_rollout_random_numbers: bool,
    search_common_random_numbers: bool,
    second_settlement_rollout_samples: usize,
    heuristic_refinement: bool,
    endgame_conversion: bool,
    prefer_city_conversion: bool,
    prefer_city_conversion_seat_mask: u8,
    immediate_vp_min: i32,
    conversion_min_vp: i32,
    proposal_conversion_min_vp: i32,
    conversion_saving_min_vp: i32,
    conversion_saving_max_deficit: i16,
    endgame_road_push: bool,
    endgame_road_push_seat_mask: u8,
    opening_road_planning: bool,
    road_refinement: bool,
    road_length_weight: f32,
    road_settlement_weight: f32,
    knight_pressure: bool,
    knight_pressure_min_vp: i32,
    knight_pressure_seat_mask: u8,
    leader_robber_weight: f32,
    blocking_settlement_weight: f32,
    trade_refinement: bool,
    resource_tactics: bool,
    end_turn_trade_sweep: bool,
    end_turn_trade_sweep_max_vp: i32,
    state_refinement_mix: f32,
    state_params: ValueParams,
    search_root_k: usize,
    search_samples: usize,
    search_continuation_decisions: usize,
    search_value_weight: f32,
    search_potential_weight: f32,
    search_vp_gain_weight: f32,
    search_building_gain_weight: f32,
    search_road_control_weight: f32,
    search_min_vp: i32,
    search_max_vp: i32,
    search_max_hidden_vp: i8,
    search_to_terminal: bool,
    search_max_decisions: usize,
    search_opponent_alpha: AlphaConfig,
    independent_rollout_seeds: bool,
    search_seed: u64,
    search_decision: u64,
    episode_search_decisions: HashMap<(u64, usize), u64>,
}

#[pymethods]
impl OpeningHybridPolicy {
    #[new]
    #[pyo3(signature = (
        net_path,
        specialist_net_path=None,
        specialist_net_min_vp=0,
        specialist_net_seat_mask=15,
        late_net_path=None,
        late_net_min_vp=5,
        late_net_seat_mask=15,
        final_net_path=None,
        final_net_min_vp=6,
        final_net_min_cities=0,
        final_net_max_cities=4,
        final_net_seat_mask=15,
        heuristic="v2",
        seed=0,
        settlement_neural_mix=0.0,
        strategy_settlement_weight=0.0,
        opening_production_weight=0.0,
        opening_wheat_weight=0.0,
        opening_wheat_seat_mask=15,
        opening_city_weight=0.0,
        opening_city_seat_mask=15,
        opening_settlement_lookahead=false,
        opening_rollout_candidates=12,
        opening_rollout_samples=0,
        opening_rollout_finalists=2,
        opening_rollout_final_samples=0,
        opening_rollout_prior_weight=0.0,
        rollout_vp_margin_weight=0.0,
        common_rollout_random_numbers=false,
        search_common_random_numbers=false,
        second_settlement_rollout_samples=0,
        heuristic_refinement=false,
        endgame_conversion=false,
        prefer_city_conversion=false,
        prefer_city_conversion_seat_mask=15,
        immediate_vp_min=5,
        conversion_min_vp=5,
        proposal_conversion_min_vp=5,
        conversion_saving_min_vp=8,
        conversion_saving_max_deficit=0,
        endgame_road_push=false,
        endgame_road_push_seat_mask=15,
        opening_road_planning=false,
        road_refinement=false,
        road_length_weight=5.0,
        road_settlement_weight=20.0,
        knight_pressure=false,
        knight_pressure_min_vp=0,
        knight_pressure_seat_mask=15,
        leader_robber_weight=0.0,
        blocking_settlement_weight=0.0,
        trade_refinement=false,
        resource_tactics=false,
        end_turn_trade_sweep=false,
        end_turn_trade_sweep_max_vp=7,
        evolved_state_refinement=false,
        state_refinement_mix=0.0,
        search_root_k=1,
        search_samples=1,
        search_continuation_decisions=0,
        search_value_weight=1.0,
        search_potential_weight=0.0,
        search_vp_gain_weight=0.0,
        search_building_gain_weight=0.0,
        search_road_control_weight=0.0,
        search_min_vp=0,
        search_max_vp=7,
        search_max_hidden_vp=5,
        search_to_terminal=false,
        search_max_decisions=64,
        search_opponent_root_k=1,
        search_opponent_samples=1,
        search_opponent_depth=0,
        independent_rollout_seeds=false
    ))]
    fn new(
        net_path: String,
        specialist_net_path: Option<String>,
        specialist_net_min_vp: i32,
        specialist_net_seat_mask: u8,
        late_net_path: Option<String>,
        late_net_min_vp: i32,
        late_net_seat_mask: u8,
        final_net_path: Option<String>,
        final_net_min_vp: i32,
        final_net_min_cities: u8,
        final_net_max_cities: u8,
        final_net_seat_mask: u8,
        heuristic: &str,
        seed: u64,
        settlement_neural_mix: f32,
        strategy_settlement_weight: f32,
        opening_production_weight: f32,
        opening_wheat_weight: f32,
        opening_wheat_seat_mask: u8,
        opening_city_weight: f32,
        opening_city_seat_mask: u8,
        opening_settlement_lookahead: bool,
        opening_rollout_candidates: usize,
        opening_rollout_samples: usize,
        opening_rollout_finalists: usize,
        opening_rollout_final_samples: usize,
        opening_rollout_prior_weight: f32,
        rollout_vp_margin_weight: f32,
        common_rollout_random_numbers: bool,
        search_common_random_numbers: bool,
        second_settlement_rollout_samples: usize,
        heuristic_refinement: bool,
        endgame_conversion: bool,
        prefer_city_conversion: bool,
        prefer_city_conversion_seat_mask: u8,
        immediate_vp_min: i32,
        conversion_min_vp: i32,
        proposal_conversion_min_vp: i32,
        conversion_saving_min_vp: i32,
        conversion_saving_max_deficit: i16,
        endgame_road_push: bool,
        endgame_road_push_seat_mask: u8,
        opening_road_planning: bool,
        road_refinement: bool,
        road_length_weight: f32,
        road_settlement_weight: f32,
        knight_pressure: bool,
        knight_pressure_min_vp: i32,
        knight_pressure_seat_mask: u8,
        leader_robber_weight: f32,
        blocking_settlement_weight: f32,
        trade_refinement: bool,
        resource_tactics: bool,
        end_turn_trade_sweep: bool,
        end_turn_trade_sweep_max_vp: i32,
        evolved_state_refinement: bool,
        state_refinement_mix: f32,
        search_root_k: usize,
        search_samples: usize,
        search_continuation_decisions: usize,
        search_value_weight: f32,
        search_potential_weight: f32,
        search_vp_gain_weight: f32,
        search_building_gain_weight: f32,
        search_road_control_weight: f32,
        search_min_vp: i32,
        search_max_vp: i32,
        search_max_hidden_vp: i8,
        search_to_terminal: bool,
        search_max_decisions: usize,
        search_opponent_root_k: usize,
        search_opponent_samples: usize,
        search_opponent_depth: u32,
        independent_rollout_seeds: bool,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&settlement_neural_mix) {
            return Err(PyValueError::new_err(
                "settlement_neural_mix must be between 0 and 1",
            ));
        }
        if strategy_settlement_weight < 0.0 {
            return Err(PyValueError::new_err(
                "strategy_settlement_weight must be non-negative",
            ));
        }
        if opening_production_weight < 0.0 {
            return Err(PyValueError::new_err(
                "opening_production_weight must be non-negative",
            ));
        }
        if opening_wheat_weight < 0.0 {
            return Err(PyValueError::new_err(
                "opening_wheat_weight must be non-negative",
            ));
        }
        if opening_wheat_seat_mask > 15 {
            return Err(PyValueError::new_err(
                "opening_wheat_seat_mask must use only the low four bits",
            ));
        }
        if opening_city_weight < 0.0 {
            return Err(PyValueError::new_err(
                "opening_city_weight must be non-negative",
            ));
        }
        if opening_city_seat_mask > 15 {
            return Err(PyValueError::new_err(
                "opening_city_seat_mask must use only the low four bits",
            ));
        }
        if prefer_city_conversion_seat_mask > 15 {
            return Err(PyValueError::new_err(
                "prefer_city_conversion_seat_mask must use only the low four bits",
            ));
        }
        if !(0..=7).contains(&knight_pressure_min_vp) {
            return Err(PyValueError::new_err(
                "knight_pressure_min_vp must be between 0 and 7",
            ));
        }
        if knight_pressure_seat_mask > 15 {
            return Err(PyValueError::new_err(
                "knight_pressure_seat_mask must use only the low four bits",
            ));
        }
        if specialist_net_min_vp < 0 {
            return Err(PyValueError::new_err(
                "specialist_net_min_vp must be non-negative",
            ));
        }
        if specialist_net_seat_mask == 0 || specialist_net_seat_mask > 15 {
            return Err(PyValueError::new_err(
                "specialist_net_seat_mask must use at least one of the low four bits",
            ));
        }
        if late_net_min_vp < 0 {
            return Err(PyValueError::new_err(
                "late_net_min_vp must be non-negative",
            ));
        }
        if late_net_seat_mask == 0 || late_net_seat_mask > 15 {
            return Err(PyValueError::new_err(
                "late_net_seat_mask must use at least one of the low four bits",
            ));
        }
        if endgame_road_push_seat_mask > 15 {
            return Err(PyValueError::new_err(
                "endgame_road_push_seat_mask must use only the low four bits",
            ));
        }
        if final_net_min_vp < 0 {
            return Err(PyValueError::new_err(
                "final_net_min_vp must be non-negative",
            ));
        }
        if final_net_min_cities > 4 {
            return Err(PyValueError::new_err(
                "final_net_min_cities must be between 0 and 4",
            ));
        }
        if final_net_max_cities > 4 || final_net_max_cities < final_net_min_cities {
            return Err(PyValueError::new_err(
                "final_net_max_cities must be between final_net_min_cities and 4",
            ));
        }
        if final_net_seat_mask == 0 || final_net_seat_mask > 15 {
            return Err(PyValueError::new_err(
                "final_net_seat_mask must use at least one of the low four bits",
            ));
        }
        if opening_rollout_candidates == 0 {
            return Err(PyValueError::new_err(
                "opening rollout candidates must be positive",
            ));
        }
        if opening_rollout_finalists == 0 {
            return Err(PyValueError::new_err(
                "opening rollout finalists must be positive",
            ));
        }
        if opening_rollout_prior_weight < 0.0 {
            return Err(PyValueError::new_err(
                "opening rollout prior weight must be non-negative",
            ));
        }
        if rollout_vp_margin_weight < 0.0 {
            return Err(PyValueError::new_err(
                "rollout_vp_margin_weight must be non-negative",
            ));
        }
        if leader_robber_weight < 0.0 {
            return Err(PyValueError::new_err(
                "leader_robber_weight must be non-negative",
            ));
        }
        if road_length_weight < 0.0
            || road_settlement_weight < 0.0
            || road_length_weight + road_settlement_weight <= 0.0
        {
            return Err(PyValueError::new_err(
                "road refinement weights must be non-negative with positive total",
            ));
        }
        if blocking_settlement_weight < 0.0 {
            return Err(PyValueError::new_err(
                "blocking_settlement_weight must be non-negative",
            ));
        }
        if !(0..=7).contains(&immediate_vp_min) {
            return Err(PyValueError::new_err(
                "immediate_vp_min must be between 0 and 7",
            ));
        }
        if !(0..=7).contains(&conversion_min_vp) {
            return Err(PyValueError::new_err(
                "conversion_min_vp must be between 0 and 7",
            ));
        }
        if !(0..=7).contains(&proposal_conversion_min_vp) {
            return Err(PyValueError::new_err(
                "proposal_conversion_min_vp must be between 0 and 7",
            ));
        }
        if !(0..=8).contains(&conversion_saving_min_vp) {
            return Err(PyValueError::new_err(
                "conversion saving minimum VP must be between 0 and 8",
            ));
        }
        if conversion_saving_max_deficit < 0 {
            return Err(PyValueError::new_err(
                "conversion saving maximum deficit must be non-negative",
            ));
        }
        if !(0.0..=1.0).contains(&state_refinement_mix) {
            return Err(PyValueError::new_err(
                "state_refinement_mix must be between 0 and 1",
            ));
        }
        if search_root_k == 0 || search_samples == 0 {
            return Err(PyValueError::new_err(
                "search_root_k and search_samples must be positive",
            ));
        }
        if search_value_weight < 0.0
            || search_potential_weight < 0.0
            || search_vp_gain_weight < 0.0
            || search_building_gain_weight < 0.0
            || search_road_control_weight < 0.0
            || search_value_weight
                + search_potential_weight
                + search_vp_gain_weight
                + search_building_gain_weight
                + search_road_control_weight
                <= 0.0
        {
            return Err(PyValueError::new_err(
                "hybrid search score weights must be non-negative with positive total",
            ));
        }
        if !(0..=7).contains(&search_min_vp) {
            return Err(PyValueError::new_err(
                "hybrid search minimum VP must be between 0 and 7",
            ));
        }
        if !(0..=7).contains(&end_turn_trade_sweep_max_vp) {
            return Err(PyValueError::new_err(
                "end-turn trade sweep maximum VP must be between 0 and 7",
            ));
        }
        if !(search_min_vp..=7).contains(&search_max_vp) {
            return Err(PyValueError::new_err(
                "hybrid search maximum VP must be between the minimum and 7",
            ));
        }
        if !(0..=5).contains(&search_max_hidden_vp) {
            return Err(PyValueError::new_err(
                "hybrid search maximum hidden VP must be between 0 and 5",
            ));
        }
        if search_max_decisions == 0 {
            return Err(PyValueError::new_err(
                "hybrid terminal search maximum decisions must be positive",
            ));
        }
        if search_opponent_root_k == 0 || search_opponent_samples == 0 {
            return Err(PyValueError::new_err(
                "search opponent root_k and samples must be positive",
            ));
        }
        let heuristic = match heuristic {
            "v1" => HeuristicPlayer::new(seed),
            "v2" => HeuristicPlayer::v2(seed),
            other => {
                return Err(PyValueError::new_err(format!(
                    "opening heuristic must be v1 or v2, got '{other}'"
                )))
            }
        };
        let state_params = if evolved_state_refinement {
            ValueParams::from_array([
                1.591509,
                2.2567153,
                1.547891,
                1.2268502,
                1.6889327,
                2.5633042,
                0.20907588,
                0.059775528,
                1.0667652,
                0.047096804,
                0.00335642,
                0.059977558,
                0.0,
                0.11244303,
                0.060603622,
                0.87400866,
                0.53917414,
            ])
        } else {
            ValueParams::default()
        };
        Ok(Self {
            net: Arc::new(MlpNet::load(net_path.as_ref())),
            specialist_net: specialist_net_path
                .map(|path| Arc::new(MlpNet::load(path.as_ref()))),
            specialist_net_min_vp,
            specialist_net_seat_mask,
            late_net: late_net_path.map(|path| Arc::new(MlpNet::load(path.as_ref()))),
            late_net_min_vp,
            late_net_seat_mask,
            final_net: final_net_path.map(|path| Arc::new(MlpNet::load(path.as_ref()))),
            final_net_min_vp,
            final_net_min_cities,
            final_net_max_cities,
            final_net_seat_mask,
            heuristic,
            settlement_neural_mix,
            strategy_settlement_weight,
            opening_production_weight,
            opening_wheat_weight,
            opening_wheat_seat_mask,
            opening_city_weight,
            opening_city_seat_mask,
            opening_settlement_lookahead,
            opening_rollout_candidates,
            opening_rollout_samples,
            opening_rollout_finalists,
            opening_rollout_final_samples,
            opening_rollout_prior_weight,
            rollout_vp_margin_weight,
            common_rollout_random_numbers,
            search_common_random_numbers,
            second_settlement_rollout_samples,
            heuristic_refinement,
            endgame_conversion,
            prefer_city_conversion,
            prefer_city_conversion_seat_mask,
            immediate_vp_min,
            conversion_min_vp,
            proposal_conversion_min_vp,
            conversion_saving_min_vp,
            conversion_saving_max_deficit,
            endgame_road_push,
            endgame_road_push_seat_mask,
            opening_road_planning,
            road_refinement,
            road_length_weight,
            road_settlement_weight,
            knight_pressure,
            knight_pressure_min_vp,
            knight_pressure_seat_mask,
            leader_robber_weight,
            blocking_settlement_weight,
            trade_refinement,
            resource_tactics,
            end_turn_trade_sweep,
            end_turn_trade_sweep_max_vp,
            state_refinement_mix,
            state_params,
            search_root_k,
            search_samples,
            search_continuation_decisions,
            search_value_weight,
            search_potential_weight,
            search_vp_gain_weight,
            search_building_gain_weight,
            search_road_control_weight,
            search_min_vp,
            search_max_vp,
            search_max_hidden_vp,
            search_to_terminal,
            search_max_decisions,
            search_opponent_alpha: AlphaConfig {
                root_k: search_opponent_root_k,
                samples: search_opponent_samples,
                depth: search_opponent_depth,
            },
            independent_rollout_seeds,
            search_seed: seed,
            search_decision: 0,
            episode_search_decisions: HashMap::new(),
        })
    }

    fn action(&mut self, env: PyRef<'_, Env>) -> PyResult<usize> {
        self.action_inner(&env.inner)
    }

    /// Select one deployed-hybrid action for every live vectorized env.
    ///
    /// VecEnv advances bot seats internally, so each environment is already
    /// stopped at its single policy-controlled seat.
    fn actions(&mut self, env: PyRef<'_, VecEnv>) -> PyResult<Vec<u32>> {
        env.inner
            .envs
            .iter()
            .map(|inner| self.action_inner(inner).map(|action| action as u32))
            .collect()
    }
}

impl OpeningHybridPolicy {
    fn next_search_seed(&mut self, env: &CatanEnv, candidate: usize) -> u64 {
        if !self.independent_rollout_seeds {
            let seed = self.search_seed ^ self.search_decision.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            self.search_decision = self.search_decision.wrapping_add(1);
            return seed;
        }
        let episode_seed = env.episode_seed();
        let decision = self
            .episode_search_decisions
            .entry((episode_seed, candidate))
            .or_insert(0);
        let seed = independent_rollout_seed(self.search_seed, episode_seed, candidate, *decision);
        *decision = decision.wrapping_add(1);
        seed
    }

    fn active_net(&self, game: &CatanGame, candidate: usize) -> Arc<MlpNet> {
        if let Some(net) = self.final_net.as_ref().filter(|_| {
            game.state.calculate_victory_points(candidate) >= self.final_net_min_vp
                && game.state.cities_built[candidate] >= self.final_net_min_cities
                && game.state.cities_built[candidate] <= self.final_net_max_cities
                && self.final_net_seat_mask & (1 << candidate) != 0
        }) {
            return net.clone();
        }
        if let Some(net) = self.late_net.as_ref().filter(|_| {
            game.state.calculate_victory_points(candidate) >= self.late_net_min_vp
                && self.late_net_seat_mask & (1 << candidate) != 0
        }) {
            return net.clone();
        }
        self.specialist_net
            .as_ref()
            .filter(|_| {
                game.state.calculate_victory_points(candidate) >= self.specialist_net_min_vp
                    && self.specialist_net_seat_mask & (1 << candidate) != 0
            })
            .cloned()
            .unwrap_or_else(|| self.net.clone())
    }

    fn action_inner(&mut self, env: &CatanEnv) -> PyResult<usize> {
        let candidate = OpponentAwarePlanner::validate_env(env)?;
        let game = env.game();
        let mut valid = Vec::with_capacity(128);
        game.fill_valid_actions(&mut valid);
        if uses_opening_heuristic(&valid) {
            let action = if self.opening_settlement_lookahead
                && game.state.settlements_built[candidate] == 0
            {
                self.lookahead_first_settlement_action(env, candidate, &valid)
            } else if self.strategy_settlement_weight > 0.0 {
                if self.second_settlement_rollout_samples > 0
                    && game.state.settlements_built[candidate] > 0
                {
                    self.rollout_settlement_action(env, candidate, &valid)
                } else {
                    self.strategy_settlement_action(game, candidate, &valid)
                }
            } else if self.settlement_neural_mix == 0.0 {
                self.heuristic.choose_action(game, &valid)
            } else {
                self.blended_settlement_action(env, &valid)
            };
            return Ok(encode_action(game, &action));
        }
        if self.opening_road_planning && uses_opening_road_planning(&valid) {
            return Ok(encode_action(
                game,
                &opening_road_action(game, candidate, &valid),
            ));
        }
        if self.endgame_conversion {
            let prefer_city_conversion = self.prefer_city_conversion
                && self.prefer_city_conversion_seat_mask & (1 << candidate) != 0;
            if let Some(action) = immediate_vp_action(
                game,
                candidate,
                &valid,
                self.immediate_vp_min,
                &self.heuristic,
                self.blocking_settlement_weight,
                prefer_city_conversion,
            ) {
                return Ok(encode_action(game, &action));
            }
            if let Some(action) = conversion_resource_action(
                game,
                candidate,
                &valid,
                self.conversion_min_vp,
                self.proposal_conversion_min_vp,
                prefer_city_conversion,
            ) {
                return Ok(encode_action(game, &action));
            }
            if self.endgame_road_push
                && self.endgame_road_push_seat_mask & (1 << candidate) != 0
            {
                if let Some(action) =
                    conversion_road_action(game, candidate, &valid, self.conversion_min_vp)
                {
                    return Ok(encode_action(game, &action));
                }
            }
        }
        if self.knight_pressure
            && game.state.calculate_victory_points(candidate) >= self.knight_pressure_min_vp
            && self.knight_pressure_seat_mask & (1 << candidate) != 0
        {
            if let Some(action) = pressure_knight_action(game, candidate, &valid) {
                return Ok(encode_action(game, &action));
            }
        }
        let candidate_vp = game.state.calculate_victory_points(candidate);
        let search_enabled = candidate_vp >= self.search_min_vp
            && candidate_vp <= self.search_max_vp
            && game.state.dev_cards[candidate][DEV_VICTORY_POINT] <= self.search_max_hidden_vp;
        if self.search_to_terminal
            && search_enabled
            && self.search_root_k > 1
            && self.search_samples > 1
        {
            let search_seed = self.next_search_seed(env, candidate);
            return Ok(self.terminal_search_action(env, candidate, search_seed));
        }
        if self.state_refinement_mix > 0.0 && game.turn_phase == TurnPhase::Main {
            return Ok(encode_action(
                game,
                &self.blended_state_action(env, candidate, &valid),
            ));
        }
        let planner = OpponentAwarePlanner {
            net: self.active_net(game, candidate),
            config: PlannerConfig {
                root_k: if search_enabled {
                    self.search_root_k
                } else {
                    1
                },
                samples: if search_enabled {
                    self.search_samples
                } else {
                    1
                },
                continuation_decisions: if search_enabled {
                    self.search_continuation_decisions
                } else {
                    0
                },
                value_weight: self.search_value_weight,
                potential_weight: self.search_potential_weight,
                vp_gain_weight: self.search_vp_gain_weight,
                building_gain_weight: self.search_building_gain_weight,
                road_control_weight: self.search_road_control_weight,
                common_random_numbers: self.search_common_random_numbers,
                opponent_alpha: AlphaConfig {
                    root_k: 1,
                    samples: 1,
                    depth: 0,
                },
            },
        };
        let search_seed = self.next_search_seed(env, candidate);
        let neural_id = planner.action_inner(env, candidate, search_seed);
        if !self.heuristic_refinement {
            return Ok(neural_id);
        }
        let neural_action = valid
            .iter()
            .find(|action| encode_action(game, action) == neural_id)
            .expect("neural action must be present in the legal set");
        if matches!(neural_action, Action::BuyDevCard { .. })
            && game.state.calculate_victory_points(candidate) >= self.conversion_saving_min_vp
            && next_build_deficit(&game.state, candidate, game.state.resources[candidate])
                .is_some_and(|deficit| deficit <= self.conversion_saving_max_deficit)
        {
            if let Some(action) = valid
                .iter()
                .copied()
                .find(|action| matches!(action, Action::EndTurn { .. }))
            {
                return Ok(encode_action(game, &action));
            }
        }
        if self.end_turn_trade_sweep
            && game.state.calculate_victory_points(candidate) <= self.end_turn_trade_sweep_max_vp
            && matches!(neural_action, Action::EndTurn { .. })
        {
            if let Some(action) = end_turn_trade_action(game, candidate, &valid) {
                return Ok(encode_action(game, &action));
            }
        }
        if self.resource_tactics
            && matches!(
                neural_action,
                Action::TradeWithBank { .. }
                    | Action::PlayYearOfPlenty { .. }
                    | Action::PlayMonopoly { .. }
            )
        {
            return Ok(encode_action(
                game,
                &refined_resource_action(game, candidate, neural_action, &valid),
            ));
        }
        if matches!(neural_action, Action::ProposeTrade { .. }) {
            if self.trade_refinement {
                return Ok(encode_action(
                    game,
                    &refined_trade_action(game, candidate, &valid),
                ));
            }
            return Ok(neural_id);
        }
        let Some(family) = refinement_family(neural_action) else {
            return Ok(neural_id);
        };
        if family == RefinementFamily::Road && self.road_refinement {
            return Ok(encode_action(
                game,
                &refined_road_action(
                    game,
                    candidate,
                    &valid,
                    self.road_length_weight,
                    self.road_settlement_weight,
                ),
            ));
        }
        if family == RefinementFamily::Discard && self.resource_tactics {
            return Ok(encode_action(
                game,
                &refined_discard_action(game, candidate, &valid),
            ));
        }
        if family == RefinementFamily::Settlement && self.blocking_settlement_weight > 0.0 {
            return Ok(encode_action(
                game,
                &refined_settlement_action(
                    game,
                    candidate,
                    &valid,
                    &self.heuristic,
                    self.blocking_settlement_weight,
                ),
            ));
        }
        if family == RefinementFamily::Robber && self.leader_robber_weight > 0.0 {
            return Ok(encode_action(
                game,
                &refined_robber_action(
                    game,
                    candidate,
                    &valid,
                    &self.heuristic,
                    self.leader_robber_weight,
                ),
            ));
        }
        let family_actions: Vec<Action> = valid
            .iter()
            .copied()
            .filter(|action| refinement_family(action) == Some(family))
            .collect();
        let refined = self.heuristic.choose_action(game, &family_actions);
        Ok(encode_action(game, &refined))
    }

    fn terminal_search_action(&self, env: &CatanEnv, candidate: usize, seed: u64) -> usize {
        let planner = OpponentAwarePlanner {
            net: self.active_net(env.game(), candidate),
            config: PlannerConfig {
                root_k: self.search_root_k,
                samples: 1,
                continuation_decisions: 0,
                value_weight: self.search_value_weight,
                potential_weight: self.search_potential_weight,
                vp_gain_weight: self.search_vp_gain_weight,
                building_gain_weight: self.search_building_gain_weight,
                road_control_weight: self.search_road_control_weight,
                common_random_numbers: self.common_rollout_random_numbers,
                opponent_alpha: AlphaConfig {
                    root_k: 1,
                    samples: 1,
                    depth: 0,
                },
            },
        };
        let actions = planner.ranked_actions(env, None);
        if actions.len() == 1 {
            return actions[0];
        }
        actions
            .par_iter()
            .enumerate()
            .map(|(action_index, action)| {
                let score = (0..self.search_samples)
                    .into_par_iter()
                    .map(|sample_index| {
                        let rollout_seed =
                            seed ^ (sample_index as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                        let rollout_seed = if self.common_rollout_random_numbers {
                            rollout_seed
                        } else {
                            rollout_seed
                                ^ (action_index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        };
                        self.hybrid_terminal_rollout(env, candidate, *action, rollout_seed, true)
                    })
                    .sum::<f32>()
                    / self.search_samples as f32;
                (score, *action)
            })
            .max_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal))
            .map(|(_, action)| action)
            .expect("terminal search requires at least one legal action")
    }

    fn hybrid_terminal_rollout(
        &self,
        base: &CatanEnv,
        candidate: usize,
        root_action: usize,
        seed: u64,
        use_final_net: bool,
    ) -> f32 {
        let opponent_alpha = if use_final_net {
            self.search_opponent_alpha
        } else {
            AlphaConfig {
                root_k: 1,
                samples: 1,
                depth: 0,
            }
        };
        let mut env = base.search_sample_with_alpha(candidate, seed, opponent_alpha);
        let mut legal = vec![false; NUM_ACTIONS];
        env.write_mask(&mut legal);
        if !legal[root_action] || !action_executes(&env, root_action) {
            return -1.0;
        }
        let mut result = env.step(root_action);
        if result.done {
            return self.hybrid_terminal_score(&env, candidate, result.winner);
        }

        let mut rollout_policy = self.clone();
        if !use_final_net {
            rollout_policy.final_net = None;
        }
        rollout_policy.search_to_terminal = false;
        rollout_policy.opening_rollout_samples = 0;
        rollout_policy.second_settlement_rollout_samples = 0;
        rollout_policy.search_root_k = 1;
        rollout_policy.search_samples = 1;
        rollout_policy.search_continuation_decisions = 0;
        rollout_policy.search_seed = seed;
        rollout_policy.search_decision = 0;
        rollout_policy.episode_search_decisions.clear();
        for _ in 0..self.search_max_decisions {
            let action = match rollout_policy.action_inner(&env) {
                Ok(action) => action,
                Err(_) => return -1.0,
            };
            if !action_executes(&env, action) {
                return -1.0;
            }
            result = env.step(action);
            if result.done {
                return self.hybrid_terminal_score(&env, candidate, result.winner);
            }
        }

        let own_vp = env.game().state.calculate_victory_points(candidate) as f32;
        let best_opponent = (0..env.config.num_players)
            .filter(|seat| *seat != candidate)
            .map(|seat| env.game().state.calculate_victory_points(seat))
            .max()
            .unwrap_or(0) as f32;
        ((own_vp - best_opponent) / env.game().state.victory_target.max(1) as f32).clamp(-1.0, 1.0)
    }

    fn hybrid_terminal_score(&self, env: &CatanEnv, candidate: usize, winner: i8) -> f32 {
        let win_score = OpponentAwarePlanner::terminal_score(candidate, winner);
        if self.rollout_vp_margin_weight == 0.0 {
            return win_score;
        }
        let own_vp = env.game().state.calculate_victory_points(candidate) as f32;
        let best_opponent = (0..env.config.num_players)
            .filter(|seat| *seat != candidate)
            .map(|seat| env.game().state.calculate_victory_points(seat))
            .max()
            .unwrap_or(0) as f32;
        let margin = (own_vp - best_opponent) / env.game().state.victory_target.max(1) as f32;
        win_score + self.rollout_vp_margin_weight * margin
    }

    fn lookahead_first_settlement_action(
        &mut self,
        env: &CatanEnv,
        candidate: usize,
        valid: &[Action],
    ) -> Action {
        let mut candidates = valid.to_vec();
        candidates.sort_by(|left, right| {
            self.heuristic
                .score_action(env.game(), right)
                .partial_cmp(&self.heuristic.score_action(env.game(), left))
                .unwrap_or(Ordering::Equal)
        });
        candidates.truncate(self.opening_rollout_candidates);
        let search_seed = self.next_search_seed(env, candidate);

        if self.opening_rollout_samples > 0 {
            let mut scored: Vec<(f32, f32, usize, Action)> = candidates
                .par_iter()
                .enumerate()
                .map(|(action_index, action)| {
                    let score = (0..self.opening_rollout_samples)
                        .into_par_iter()
                        .map(|sample_index| {
                            let rollout_seed = search_seed
                                ^ (sample_index as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                            let rollout_seed = if self.common_rollout_random_numbers {
                                rollout_seed
                            } else {
                                rollout_seed
                                    ^ (action_index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            };
                            self.hybrid_terminal_rollout(
                                env,
                                candidate,
                                encode_action(env.game(), action),
                                rollout_seed,
                                false,
                            )
                        })
                        .sum::<f32>()
                        / self.opening_rollout_samples as f32;
                    let prior =
                        self.first_settlement_pair_score(env, candidate, *action, search_seed);
                    (score, prior, action_index, *action)
                })
                .collect();
            let combined_score = |score: f32, prior: f32| {
                score + self.opening_rollout_prior_weight * prior + 1e-6 * prior
            };
            scored.sort_by(|left, right| {
                combined_score(right.0, right.1)
                    .partial_cmp(&combined_score(left.0, left.1))
                    .unwrap_or(Ordering::Equal)
            });
            if self.opening_rollout_final_samples == 0 {
                return scored[0].3;
            }
            scored.truncate(self.opening_rollout_finalists.min(scored.len()));
            return scored
                .par_iter()
                .map(|(initial_score, prior, action_index, action)| {
                    let extra_sum = (0..self.opening_rollout_final_samples)
                        .into_par_iter()
                        .map(|extra_index| {
                            let sample_index = self.opening_rollout_samples + extra_index;
                            let rollout_seed = search_seed
                                ^ (sample_index as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                            let rollout_seed = if self.common_rollout_random_numbers {
                                rollout_seed
                            } else {
                                rollout_seed
                                    ^ (*action_index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            };
                            self.hybrid_terminal_rollout(
                                env,
                                candidate,
                                encode_action(env.game(), action),
                                rollout_seed,
                                false,
                            )
                        })
                        .sum::<f32>();
                    let total_samples =
                        (self.opening_rollout_samples + self.opening_rollout_final_samples) as f32;
                    let score = (*initial_score * self.opening_rollout_samples as f32 + extra_sum)
                        / total_samples;
                    (combined_score(score, *prior), *action)
                })
                .max_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal))
                .map(|(_, action)| action)
                .expect("opening rollout finalists cannot be empty");
        }

        candidates
            .into_iter()
            .max_by(|left, right| {
                self.first_settlement_pair_score(env, candidate, *left, search_seed)
                    .partial_cmp(&self.first_settlement_pair_score(
                        env,
                        candidate,
                        *right,
                        search_seed,
                    ))
                    .unwrap_or(Ordering::Equal)
            })
            .expect("first settlement lookahead requires legal actions")
    }

    fn first_settlement_pair_score(
        &self,
        env: &CatanEnv,
        candidate: usize,
        action: Action,
        search_seed: u64,
    ) -> f32 {
        let mut sample = env.search_sample_with_alpha(
            candidate,
            search_seed,
            AlphaConfig {
                root_k: 1,
                samples: 1,
                depth: 0,
            },
        );
        let first_id = encode_action(sample.game(), &action);
        let result = sample.step(first_id);
        if result.done || sample.current_seat() != candidate {
            return f32::NEG_INFINITY;
        }
        let mut roads = Vec::with_capacity(3);
        sample.game().fill_valid_actions(&mut roads);
        if !roads
            .iter()
            .all(|road| matches!(road, Action::PlaceInitialRoad { .. }))
        {
            return f32::NEG_INFINITY;
        }
        let planner = OpponentAwarePlanner {
            net: self.net.clone(),
            config: PlannerConfig {
                root_k: 1,
                samples: 1,
                continuation_decisions: 0,
                value_weight: self.search_value_weight,
                potential_weight: self.search_potential_weight,
                vp_gain_weight: 0.0,
                building_gain_weight: 0.0,
                road_control_weight: 0.0,
                common_random_numbers: self.common_rollout_random_numbers,
                opponent_alpha: AlphaConfig {
                    root_k: 1,
                    samples: 1,
                    depth: 0,
                },
            },
        };
        let road_id = planner.action_inner(&sample, candidate, search_seed);
        let result = sample.step(road_id);
        if result.done || sample.current_seat() != candidate {
            return f32::NEG_INFINITY;
        }
        let mut second_settlements = Vec::with_capacity(54);
        sample.game().fill_valid_actions(&mut second_settlements);
        second_settlements
            .iter()
            .map(|second| self.strategy_settlement_score(sample.game(), candidate, second))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    fn strategy_settlement_action(
        &self,
        game: &catan_core::game::CatanGame,
        candidate: usize,
        valid: &[Action],
    ) -> Action {
        valid
            .iter()
            .copied()
            .max_by(|left, right| {
                self.strategy_settlement_score(game, candidate, left)
                    .partial_cmp(&self.strategy_settlement_score(game, candidate, right))
                    .unwrap_or(Ordering::Equal)
            })
            .expect("setup settlement must have legal actions")
    }

    fn rollout_settlement_action(
        &mut self,
        env: &CatanEnv,
        candidate: usize,
        valid: &[Action],
    ) -> Action {
        let mut candidates = valid.to_vec();
        candidates.sort_by(|left, right| {
            self.strategy_settlement_score(env.game(), candidate, right)
                .partial_cmp(&self.strategy_settlement_score(env.game(), candidate, left))
                .unwrap_or(Ordering::Equal)
        });
        candidates.truncate(self.opening_rollout_candidates);
        let search_seed = self.next_search_seed(env, candidate);

        candidates
            .par_iter()
            .enumerate()
            .map(|(action_index, action)| {
                let score = (0..self.second_settlement_rollout_samples)
                    .into_par_iter()
                    .map(|sample_index| {
                        let rollout_seed = search_seed
                            ^ (sample_index as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                        let rollout_seed = if self.common_rollout_random_numbers {
                            rollout_seed
                        } else {
                            rollout_seed
                                ^ (action_index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        };
                        self.hybrid_terminal_rollout(
                            env,
                            candidate,
                            encode_action(env.game(), action),
                            rollout_seed,
                            false,
                        )
                    })
                    .sum::<f32>()
                    / self.second_settlement_rollout_samples as f32;
                let tie_break =
                    self.strategy_settlement_score(env.game(), candidate, action) * 1e-4;
                (score + tie_break, *action)
            })
            .max_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal))
            .map(|(_, action)| action)
            .expect("second-settlement rollout requires legal candidates")
    }

    fn strategy_settlement_score(
        &self,
        game: &catan_core::game::CatanGame,
        candidate: usize,
        action: &Action,
    ) -> f32 {
        let Action::PlaceInitialSettlement { vertex, .. } = *action else {
            return f32::NEG_INFINITY;
        };
        let heuristic_score = self.heuristic.score_action(game, action);
        if game.state.settlements_built[candidate] == 0 {
            return heuristic_score;
        }
        let mut income = income_by_resource(&game.state, candidate);
        let topo = topology();
        for tile in topo.vertex_tiles[vertex as usize] {
            if tile < 0 {
                continue;
            }
            let tile = tile as usize;
            let resource = game.state.tile_resources[tile];
            if (0..5).contains(&resource) {
                income[resource as usize] +=
                    topo.number_probabilities[game.state.tile_numbers[tile] as usize];
            }
        }
        let total_income = income.iter().sum::<f32>();
        let city_weight = if self.opening_city_seat_mask & (1 << candidate) != 0 {
            self.opening_city_weight
        } else {
            0.0
        };
        let wheat_weight = if self.opening_wheat_seat_mask & (1 << candidate) != 0 {
            self.opening_wheat_weight
        } else {
            0.0
        };
        heuristic_score
            + self.strategy_settlement_weight * coherent_resource_engine(income)
            + self.opening_production_weight * total_income
            + wheat_weight * income[0]
            + city_weight * city_resource_engine(income)
    }

    fn blended_settlement_action(&self, env: &CatanEnv, valid: &[Action]) -> Action {
        let game = env.game();
        let mut obs = vec![0.0; OBS_DIM];
        env.clone().write_obs(&mut obs);
        let mut scratch = NetScratch::new(&self.net);
        self.net.trunk(&obs, &mut scratch);

        let mut heuristic_order: Vec<usize> = (0..valid.len()).collect();
        heuristic_order.sort_by(|left, right| {
            self.heuristic
                .score_action(game, &valid[*left])
                .partial_cmp(&self.heuristic.score_action(game, &valid[*right]))
                .unwrap_or(Ordering::Equal)
        });
        let mut neural_order: Vec<usize> = (0..valid.len()).collect();
        neural_order.sort_by(|left, right| {
            let left_id = encode_action(game, &valid[*left]);
            let right_id = encode_action(game, &valid[*right]);
            self.net
                .logit_from(&scratch, left_id)
                .partial_cmp(&self.net.logit_from(&scratch, right_id))
                .unwrap_or(Ordering::Equal)
        });

        let mut heuristic_rank = vec![0.0f32; valid.len()];
        let mut neural_rank = vec![0.0f32; valid.len()];
        let denominator = valid.len().saturating_sub(1).max(1) as f32;
        for (rank, index) in heuristic_order.into_iter().enumerate() {
            heuristic_rank[index] = rank as f32 / denominator;
        }
        for (rank, index) in neural_order.into_iter().enumerate() {
            neural_rank[index] = rank as f32 / denominator;
        }

        let neural_mix = self.settlement_neural_mix;
        let best = (0..valid.len())
            .max_by(|left, right| {
                let left_score =
                    (1.0 - neural_mix) * heuristic_rank[*left] + neural_mix * neural_rank[*left];
                let right_score =
                    (1.0 - neural_mix) * heuristic_rank[*right] + neural_mix * neural_rank[*right];
                left_score
                    .partial_cmp(&right_score)
                    .unwrap_or(Ordering::Equal)
            })
            .expect("setup settlement must have legal actions");
        valid[best]
    }

    fn blended_state_action(&self, env: &CatanEnv, candidate: usize, valid: &[Action]) -> Action {
        let game = env.game();
        let mut obs = vec![0.0; OBS_DIM];
        env.clone().write_obs(&mut obs);
        let net = self.active_net(game, candidate);
        let mut scratch = NetScratch::new(&net);
        net.trunk(&obs, &mut scratch);
        let params = self.state_params;

        let mut neural_order: Vec<usize> = (0..valid.len()).collect();
        neural_order.sort_by(|left, right| {
            let left_id = encode_action(game, &valid[*left]);
            let right_id = encode_action(game, &valid[*right]);
            net.logit_from(&scratch, left_id)
                .partial_cmp(&net.logit_from(&scratch, right_id))
                .unwrap_or(Ordering::Equal)
        });
        let mut state_order: Vec<usize> = (0..valid.len()).collect();
        state_order.sort_by(|left, right| {
            state_action_score(game, candidate, &valid[*left], &params)
                .partial_cmp(&state_action_score(
                    game,
                    candidate,
                    &valid[*right],
                    &params,
                ))
                .unwrap_or(Ordering::Equal)
        });

        let mut neural_rank = vec![0.0f32; valid.len()];
        let mut state_rank = vec![0.0f32; valid.len()];
        let denominator = valid.len().saturating_sub(1).max(1) as f32;
        for (rank, index) in neural_order.into_iter().enumerate() {
            neural_rank[index] = rank as f32 / denominator;
        }
        for (rank, index) in state_order.into_iter().enumerate() {
            state_rank[index] = rank as f32 / denominator;
        }

        let state_mix = self.state_refinement_mix;
        let best = (0..valid.len())
            .max_by(|left, right| {
                let left_score =
                    (1.0 - state_mix) * neural_rank[*left] + state_mix * state_rank[*left];
                let right_score =
                    (1.0 - state_mix) * neural_rank[*right] + state_mix * state_rank[*right];
                left_score
                    .partial_cmp(&right_score)
                    .unwrap_or(Ordering::Equal)
            })
            .expect("main-phase decision must have legal actions");
        valid[best]
    }
}

fn independent_rollout_seed(
    search_seed: u64,
    episode_seed: u64,
    candidate: usize,
    decision: u64,
) -> u64 {
    search_seed
        ^ episode_seed.wrapping_mul(0xD6E8_FEB8_6659_FD93)
        ^ (candidate as u64 + 1).wrapping_mul(0xA076_1D64_78BD_642F)
        ^ decision.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn coherent_resource_engine(income: [f32; 5]) -> f32 {
    let [wheat, sheep, wood, brick, stone] = income;
    let expansion = (wood / 2.0).min(brick / 2.0).min(wheat).min(sheep);
    let city = city_resource_engine(income);
    let development = wheat.min(sheep).min(stone);
    expansion.max(city).max(development)
}

fn city_resource_engine(income: [f32; 5]) -> f32 {
    let [wheat, _, _, _, stone] = income;
    (wheat / 2.0).min(stone / 3.0)
}

fn immediate_vp_action(
    game: &catan_core::game::CatanGame,
    candidate: usize,
    valid: &[Action],
    conversion_min_vp: i32,
    heuristic: &HeuristicPlayer,
    blocking_settlement_weight: f32,
    prefer_city: bool,
) -> Option<Action> {
    let current_vp = game.state.calculate_victory_points(candidate);
    if current_vp < conversion_min_vp {
        return None;
    }
    let actions: Vec<(i32, Action)> = valid
        .iter()
        .copied()
        .filter(|action| {
            matches!(
                action,
                Action::BuildRoad { .. }
                    | Action::BuildSettlement { .. }
                    | Action::BuildCity { .. }
                    | Action::PlayKnight { .. }
            )
        })
        .filter_map(|action| {
            let mut next = game.clone();
            next.execute_action(&action).then(|| {
                (
                    next.state.calculate_victory_points(candidate) - current_vp,
                    action,
                )
            })
        })
        .filter(|(vp_gain, _)| *vp_gain > 0)
        .collect();
    let baseline = actions
        .iter()
        .copied()
        .max_by_key(|(vp_gain, action)| {
            (
                *vp_gain,
                i32::from(prefer_city && matches!(action, Action::BuildCity { .. })),
            )
        })
        .map(|(_, action)| action)?;
    if blocking_settlement_weight <= 0.0 || !matches!(baseline, Action::BuildSettlement { .. }) {
        return Some(baseline);
    }
    let max_gain = actions
        .iter()
        .map(|(vp_gain, _)| *vp_gain)
        .max()
        .expect("immediate VP candidates cannot be empty");
    actions
        .into_iter()
        .filter(|(vp_gain, action)| {
            *vp_gain == max_gain && matches!(action, Action::BuildSettlement { .. })
        })
        .max_by(|left, right| {
            let score = |action: &Action| {
                heuristic.score_action(game, action)
                    + blocking_settlement_weight * settlement_denial_score(game, candidate, action)
            };
            score(&left.1)
                .partial_cmp(&score(&right.1))
                .unwrap_or(Ordering::Equal)
        })
        .map(|(_, action)| action)
}

fn cost_deficit(resources: [i16; 5], cost: [i16; 5]) -> i16 {
    resources
        .iter()
        .zip(cost)
        .map(|(have, need)| (need - have).max(0))
        .sum()
}

fn has_settlement_spot(state: &catan_core::state::GameState, player: usize) -> bool {
    if state.settlements_built[player] >= state.max_settlements {
        return false;
    }
    let topo = topology();
    let mut spots = state.vertex_road_mask[player] & !state.occupied_mask & ((1u64 << 54) - 1);
    while spots != 0 {
        let vertex = spots.trailing_zeros() as usize;
        spots &= spots - 1;
        if state.occupied_mask & topo.neighbor_mask[vertex] == 0 {
            return true;
        }
    }
    false
}

fn next_build_deficit(
    state: &catan_core::state::GameState,
    candidate: usize,
    resources: [i16; 5],
) -> Option<i16> {
    let city = (state.settlements_built[candidate] > 0
        && state.cities_built[candidate] < state.max_cities)
        .then(|| cost_deficit(resources, CITY_COST));
    let settlement =
        has_settlement_spot(state, candidate).then(|| cost_deficit(resources, SETTLEMENT_COST));
    city.into_iter().chain(settlement).min()
}

fn conversion_build_deficit(
    state: &catan_core::state::GameState,
    candidate: usize,
    resources: [i16; 5],
    prefer_city: bool,
) -> Option<i16> {
    if prefer_city
        && state.settlements_built[candidate] > 0
        && state.cities_built[candidate] < state.max_cities
    {
        return Some(cost_deficit(resources, CITY_COST));
    }
    next_build_deficit(state, candidate, resources)
}

fn conversion_resource_action(
    game: &catan_core::game::CatanGame,
    candidate: usize,
    valid: &[Action],
    conversion_min_vp: i32,
    proposal_conversion_min_vp: i32,
    prefer_city: bool,
) -> Option<Action> {
    let candidate_vp = game.state.calculate_victory_points(candidate);
    if candidate_vp < conversion_min_vp {
        return None;
    }
    let resources = game.state.resources[candidate];
    let current_deficit =
        conversion_build_deficit(&game.state, candidate, resources, prefer_city)?;
    if current_deficit == 0 {
        return None;
    }
    valid
        .iter()
        .copied()
        .filter_map(|action| {
            let next_resources = match action {
                Action::TradeWithBank { .. } | Action::PlayYearOfPlenty { .. } => {
                    let mut next = game.clone();
                    next.execute_action(&action)
                        .then_some(next.state.resources[candidate])?
                }
                Action::ProposeTrade {
                    player,
                    give,
                    give_amount,
                    recv,
                } if player as usize == candidate && candidate_vp >= proposal_conversion_min_vp => {
                    let mut projected = resources;
                    projected[give as usize] -= give_amount as i16;
                    projected[recv as usize] += 1;
                    projected
                }
                _ => return None,
            };
            let deficit =
                conversion_build_deficit(&game.state, candidate, next_resources, prefer_city)?;
            (deficit < current_deficit).then_some((deficit, action))
        })
        .min_by_key(|(deficit, _)| *deficit)
        .map(|(_, action)| action)
}

fn conversion_road_action(
    game: &catan_core::game::CatanGame,
    candidate: usize,
    valid: &[Action],
    conversion_min_vp: i32,
) -> Option<Action> {
    if game.state.calculate_victory_points(candidate) < conversion_min_vp
        || game.state.longest_road_player == candidate as i8
        || game.state.road_lengths[candidate] < 3
    {
        return None;
    }
    if valid
        .iter()
        .any(|action| matches!(action, Action::PlayRoadBuilding { .. }))
    {
        return valid
            .iter()
            .copied()
            .find(|action| matches!(action, Action::PlayRoadBuilding { .. }));
    }
    let current_length = game.state.road_lengths[candidate];
    valid
        .iter()
        .copied()
        .filter(|action| matches!(action, Action::BuildRoad { .. }))
        .filter_map(|action| {
            let mut next = game.clone();
            next.execute_action(&action)
                .then_some((next.state.road_lengths[candidate], action))
        })
        .filter(|(length, _)| *length > current_length)
        .max_by_key(|(length, _)| *length)
        .map(|(_, action)| action)
}

fn pressure_knight_action(
    game: &catan_core::game::CatanGame,
    candidate: usize,
    valid: &[Action],
) -> Option<Action> {
    let knight = valid
        .iter()
        .copied()
        .find(|action| matches!(action, Action::PlayKnight { .. }))?;
    if game.state.largest_army_player != candidate as i8 {
        return Some(knight);
    }
    let strongest_opponent = (0..game.state.num_players)
        .filter(|player| *player != candidate)
        .map(|player| game.state.knights_played[player])
        .max()
        .unwrap_or(0);
    (strongest_opponent + 1 >= game.state.knights_played[candidate]).then_some(knight)
}

fn search_potential(game: &CatanGame, player: usize) -> f32 {
    let state = &game.state;
    let strongest_army = (0..state.num_players)
        .filter(|opponent| *opponent != player)
        .map(|opponent| state.knights_played[opponent])
        .max()
        .unwrap_or(0);
    let army_target = 3.max(strongest_army + 1);
    let army_progress = if state.largest_army_player == player as i8 {
        1.0
    } else {
        (state.knights_played[player] as f32 / army_target as f32).clamp(0.0, 1.0)
    };

    let strongest_road = (0..state.num_players)
        .filter(|opponent| *opponent != player)
        .map(|opponent| state.road_lengths[opponent])
        .max()
        .unwrap_or(0);
    let road_target = 5.max(strongest_road + 1);
    let road_progress = if state.longest_road_player == player as i8 {
        1.0
    } else {
        (state.road_lengths[player] as f32 / road_target as f32).clamp(0.0, 1.0)
    };

    0.96 * opinionated_potential(game, player) + 0.02 * army_progress + 0.02 * road_progress
}

fn state_action_score(
    game: &catan_core::game::CatanGame,
    candidate: usize,
    action: &Action,
    params: &ValueParams,
) -> f32 {
    let baseline = evaluate_state(&game.state, candidate, params);
    match action {
        Action::RollDice { .. }
        | Action::StealResource { .. }
        | Action::PlayMonopoly { .. }
        | Action::ProposeTrade { .. }
        | Action::RespondTrade { .. }
        | Action::ConfirmTrade { .. }
        | Action::EndTurn { .. } => baseline,
        Action::BuyDevCard { .. } => {
            let mut state = game.state.clone();
            for (resource, cost) in state.resources[candidate].iter_mut().zip(DEV_CARD_COST) {
                *resource -= cost;
            }
            // Expected non-VP utility plus 5/25 chance of a hidden VP card.
            evaluate_state(&state, candidate, params) + params.dev_card + 0.2 * params.vp
        }
        _ => {
            let mut next = game.clone();
            if !next.execute_action(action) {
                return f32::NEG_INFINITY;
            }
            if next.winner() == candidate as i8 {
                1000.0
            } else {
                evaluate_state(&next.state, candidate, params)
            }
        }
    }
}

fn uses_opening_heuristic(valid: &[Action]) -> bool {
    !valid.is_empty()
        && valid
            .iter()
            .all(|action| matches!(action, Action::PlaceInitialSettlement { .. }))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RefinementFamily {
    Settlement,
    City,
    Road,
    Robber,
    Steal,
    Discard,
    TradeResponse,
}

fn refinement_family(action: &Action) -> Option<RefinementFamily> {
    match action {
        Action::BuildSettlement { .. } => Some(RefinementFamily::Settlement),
        Action::BuildCity { .. } => Some(RefinementFamily::City),
        Action::BuildRoad { .. } => Some(RefinementFamily::Road),
        Action::MoveRobber { .. } => Some(RefinementFamily::Robber),
        Action::StealResource { .. } => Some(RefinementFamily::Steal),
        Action::DiscardResource { .. } => Some(RefinementFamily::Discard),
        Action::RespondTrade { .. } => Some(RefinementFamily::TradeResponse),
        _ => None,
    }
}

fn uses_opening_road_planning(valid: &[Action]) -> bool {
    !valid.is_empty()
        && valid
            .iter()
            .all(|action| matches!(action, Action::PlaceInitialRoad { .. }))
}

fn legal_settlement_vertex(
    state: &catan_core::state::GameState,
    candidate: usize,
    vertex: usize,
) -> bool {
    let bit = 1u64 << vertex;
    state.settlements_built[candidate] < state.max_settlements
        && state.occupied_mask & bit == 0
        && state.occupied_mask & topology().neighbor_mask[vertex] == 0
}

fn opening_road_action(
    game: &catan_core::game::CatanGame,
    candidate: usize,
    valid: &[Action],
) -> Action {
    let topo = topology();
    valid
        .iter()
        .copied()
        .max_by(|left, right| {
            let score = |action: &Action| {
                let Action::PlaceInitialRoad { edge, .. } = *action else {
                    return f32::NEG_INFINITY;
                };
                let endpoints = topo.edge_vertices[edge as usize];
                let frontier = endpoints
                    .into_iter()
                    .find(|vertex| game.state.occupied_mask & (1u64 << *vertex) == 0)
                    .expect("setup road must touch the just-placed settlement");
                topo.vertex_neighbors[frontier as usize]
                    .iter()
                    .copied()
                    .filter(|vertex| *vertex >= 0)
                    .map(|vertex| vertex as usize)
                    .filter(|vertex| legal_settlement_vertex(&game.state, candidate, *vertex))
                    .map(|vertex| game.state.vertex_probability(vertex))
                    .fold(0.0f32, f32::max)
            };
            score(left)
                .partial_cmp(&score(right))
                .unwrap_or(Ordering::Equal)
        })
        .expect("setup road phase must have legal actions")
}

fn best_reachable_settlement_score(state: &catan_core::state::GameState, candidate: usize) -> f32 {
    let mut spots = state.vertex_road_mask[candidate] & !state.occupied_mask;
    let mut best = 0.0f32;
    while spots != 0 {
        let vertex = spots.trailing_zeros() as usize;
        spots &= spots - 1;
        if legal_settlement_vertex(state, candidate, vertex) {
            best = best.max(state.vertex_probability(vertex));
        }
    }
    best
}

fn refined_road_action(
    game: &catan_core::game::CatanGame,
    candidate: usize,
    valid: &[Action],
    road_length_weight: f32,
    settlement_weight: f32,
) -> Action {
    let current_vp = game.state.calculate_victory_points(candidate);
    let current_road_length = game.state.road_lengths[candidate];
    valid
        .iter()
        .copied()
        .filter(|action| matches!(action, Action::BuildRoad { .. }))
        .max_by(|left, right| {
            let score = |action: &Action| {
                let mut next = game.clone();
                if !next.execute_action(action) {
                    return f32::NEG_INFINITY;
                }
                let vp_gain = next.state.calculate_victory_points(candidate) - current_vp;
                let road_gain = next.state.road_lengths[candidate] - current_road_length;
                100.0 * vp_gain as f32
                    + road_length_weight * road_gain as f32
                    + settlement_weight * best_reachable_settlement_score(&next.state, candidate)
            };
            score(left)
                .partial_cmp(&score(right))
                .unwrap_or(Ordering::Equal)
        })
        .expect("road refinement requires legal road actions")
}

fn public_vp(state: &catan_core::state::GameState, player: usize) -> i32 {
    state.settlements_built[player] as i32
        + 2 * state.cities_built[player] as i32
        + 2 * i32::from(state.longest_road_player == player as i8)
        + 2 * i32::from(state.largest_army_player == player as i8)
}

fn refined_robber_action(
    game: &catan_core::game::CatanGame,
    candidate: usize,
    valid: &[Action],
    heuristic: &HeuristicPlayer,
    leader_weight: f32,
) -> Action {
    let topo = topology();
    valid
        .iter()
        .copied()
        .filter(|action| matches!(action, Action::MoveRobber { .. }))
        .max_by(|left, right| {
            let score = |action: &Action| {
                let Action::MoveRobber { tile, .. } = *action else {
                    return f32::NEG_INFINITY;
                };
                let leader_pressure = topo.tile_vertices[tile as usize]
                    .iter()
                    .map(|vertex| game.state.settlement_owner(*vertex as usize))
                    .filter(|owner| *owner >= 0 && *owner as usize != candidate)
                    .map(|owner| public_vp(&game.state, owner as usize) as f32)
                    .sum::<f32>();
                heuristic.score_action(game, action) + leader_weight * leader_pressure
            };
            score(left)
                .partial_cmp(&score(right))
                .unwrap_or(Ordering::Equal)
        })
        .expect("robber refinement requires legal robber actions")
}

fn refined_settlement_action(
    game: &catan_core::game::CatanGame,
    candidate: usize,
    valid: &[Action],
    heuristic: &HeuristicPlayer,
    blocking_weight: f32,
) -> Action {
    valid
        .iter()
        .copied()
        .filter(|action| matches!(action, Action::BuildSettlement { .. }))
        .max_by(|left, right| {
            let score = |action: &Action| {
                let denial = settlement_denial_score(game, candidate, action);
                heuristic.score_action(game, action) + blocking_weight * denial
            };
            score(left)
                .partial_cmp(&score(right))
                .unwrap_or(Ordering::Equal)
        })
        .expect("settlement refinement requires legal settlement actions")
}

fn settlement_denial_score(
    game: &catan_core::game::CatanGame,
    candidate: usize,
    action: &Action,
) -> f32 {
    let mut next = game.clone();
    if !next.execute_action(action) {
        return f32::NEG_INFINITY;
    }
    (0..game.state.num_players)
        .filter(|player| *player != candidate)
        .map(|player| {
            let threat = 1.0 + public_vp(&game.state, player) as f32 / 7.0;
            let road_cut =
                (game.state.road_lengths[player] - next.state.road_lengths[player]).max(0) as f32;
            let expansion_cut = (best_reachable_settlement_score(&game.state, player)
                - best_reachable_settlement_score(&next.state, player))
            .max(0.0);
            threat * (0.1 * road_cut + expansion_cut)
        })
        .sum()
}

fn progress_deficit(
    state: &catan_core::state::GameState,
    candidate: usize,
    resources: [i16; 5],
) -> i16 {
    let build = next_build_deficit(state, candidate, resources).unwrap_or(i16::MAX);
    build.min(cost_deficit(resources, DEV_CARD_COST))
}

fn expected_trade_supply(
    game: &catan_core::game::CatanGame,
    candidate: usize,
    resource: usize,
) -> f32 {
    (0..game.state.num_players)
        .filter(|player| *player != candidate)
        .map(|player| income_by_resource(&game.state, player)[resource])
        .fold(0.0f32, f32::max)
}

fn refined_trade_action(
    game: &catan_core::game::CatanGame,
    candidate: usize,
    valid: &[Action],
) -> Action {
    let resources = game.state.resources[candidate];
    let current_deficit = progress_deficit(&game.state, candidate, resources);
    valid
        .iter()
        .copied()
        .filter(|action| matches!(action, Action::ProposeTrade { .. }))
        .max_by(|left, right| {
            let score = |action: &Action| {
                let Action::ProposeTrade {
                    give,
                    give_amount,
                    recv,
                    ..
                } = *action
                else {
                    return f32::NEG_INFINITY;
                };
                let mut projected = resources;
                projected[give as usize] -= give_amount as i16;
                projected[recv as usize] += 1;
                let deficit_gain =
                    current_deficit - progress_deficit(&game.state, candidate, projected);
                10.0 * deficit_gain as f32
                    + 5.0 * expected_trade_supply(game, candidate, recv as usize)
                    - 0.25 * give_amount as f32
            };
            score(left)
                .partial_cmp(&score(right))
                .unwrap_or(Ordering::Equal)
        })
        .expect("trade refinement requires legal trade proposals")
}

fn trade_progress_score(
    game: &catan_core::game::CatanGame,
    candidate: usize,
    action: Action,
) -> Option<f32> {
    let Action::ProposeTrade {
        give,
        give_amount,
        recv,
        ..
    } = action
    else {
        return None;
    };
    let resources = game.state.resources[candidate];
    let current_deficit = progress_deficit(&game.state, candidate, resources);
    let mut projected = resources;
    projected[give as usize] -= give_amount as i16;
    projected[recv as usize] += 1;
    let deficit_gain = current_deficit - progress_deficit(&game.state, candidate, projected);
    (deficit_gain > 0).then(|| {
        10.0 * deficit_gain as f32 + 5.0 * expected_trade_supply(game, candidate, recv as usize)
            - give_amount as f32
    })
}

fn end_turn_trade_action(
    game: &catan_core::game::CatanGame,
    candidate: usize,
    valid: &[Action],
) -> Option<Action> {
    let mut proposals: Vec<(f32, Action)> = valid
        .iter()
        .copied()
        .filter_map(|action| {
            trade_progress_score(game, candidate, action).map(|score| (score, action))
        })
        .collect();
    proposals.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(Ordering::Equal));
    proposals.dedup_by_key(|(_, action)| match *action {
        Action::ProposeTrade {
            give,
            give_amount,
            recv,
            ..
        } => (give, give_amount, recv),
        _ => unreachable!(),
    });
    proposals
        .get(game.trades_proposed_this_turn as usize)
        .map(|(_, action)| *action)
}

fn refined_resource_action(
    game: &catan_core::game::CatanGame,
    candidate: usize,
    neural_action: &Action,
    valid: &[Action],
) -> Action {
    let same_family = |action: &Action| {
        matches!(
            (neural_action, action),
            (Action::TradeWithBank { .. }, Action::TradeWithBank { .. })
                | (
                    Action::PlayYearOfPlenty { .. },
                    Action::PlayYearOfPlenty { .. }
                )
                | (Action::PlayMonopoly { .. }, Action::PlayMonopoly { .. })
        )
    };
    valid
        .iter()
        .copied()
        .filter(same_family)
        .max_by(|left, right| {
            let score = |action: &Action| match *action {
                Action::PlayMonopoly { resource, .. } => {
                    let shortage = DEV_CARD_COST[resource as usize]
                        - game.state.resources[candidate][resource as usize];
                    5.0 * expected_trade_supply(game, candidate, resource as usize)
                        + shortage.max(0) as f32
                }
                _ => {
                    let mut next = game.clone();
                    if !next.execute_action(action) {
                        return f32::NEG_INFINITY;
                    }
                    -10.0
                        * progress_deficit(&next.state, candidate, next.state.resources[candidate])
                            as f32
                }
            };
            score(left)
                .partial_cmp(&score(right))
                .unwrap_or(Ordering::Equal)
        })
        .expect("resource refinement requires same-family legal actions")
}

fn refined_discard_action(
    game: &catan_core::game::CatanGame,
    candidate: usize,
    valid: &[Action],
) -> Action {
    valid
        .iter()
        .copied()
        .filter(|action| matches!(action, Action::DiscardResource { .. }))
        .max_by(|left, right| {
            let score = |action: &Action| {
                let Action::DiscardResource { resource, .. } = *action else {
                    return f32::NEG_INFINITY;
                };
                let mut projected = game.state.resources[candidate];
                projected[resource as usize] -= 1;
                -10.0 * progress_deficit(&game.state, candidate, projected) as f32
                    + 0.01 * game.state.resources[candidate][resource as usize] as f32
            };
            score(left)
                .partial_cmp(&score(right))
                .unwrap_or(Ordering::Equal)
        })
        .expect("discard refinement requires legal discard actions")
}

impl OpponentAwarePlanner {
    fn validate_env(env: &CatanEnv) -> PyResult<usize> {
        if env.is_done() {
            return Err(PyValueError::new_err(
                "planner action on terminal environment",
            ));
        }
        let policy_seats: Vec<usize> = env.config.seat_kinds[..env.config.num_players]
            .iter()
            .enumerate()
            .filter_map(|(seat, kind)| (*kind == SeatKind::Policy).then_some(seat))
            .collect();
        if policy_seats.len() != 1 {
            return Err(PyValueError::new_err(
                "opponent-aware planning requires exactly one policy seat",
            ));
        }
        let candidate = policy_seats[0];
        if env.current_seat() != candidate {
            return Err(PyValueError::new_err(
                "planner can only act at the policy-controlled seat",
            ));
        }
        Ok(candidate)
    }

    fn scoring_states(
        env: &VecEnv,
        indices: Vec<usize>,
        seeds: Vec<u64>,
        required_actions: Vec<usize>,
    ) -> PyResult<Vec<ScoringState>> {
        if indices.len() != seeds.len() || indices.len() != required_actions.len() {
            return Err(PyValueError::new_err(
                "indices, seeds, and required_actions must have the same length",
            ));
        }
        let mut states = Vec::with_capacity(indices.len());
        for ((index, seed), required_action) in indices.into_iter().zip(seeds).zip(required_actions)
        {
            let base = env
                .inner
                .envs
                .get(index)
                .ok_or_else(|| PyValueError::new_err("environment index out of range"))?
                .clone();
            let candidate = Self::validate_env(&base)?;
            let mut mask = vec![false; NUM_ACTIONS];
            base.clone().write_mask(&mut mask);
            if required_action >= NUM_ACTIONS || !mask[required_action] {
                return Err(PyValueError::new_err(
                    "required action must be legal in its environment",
                ));
            }
            states.push((base, candidate, seed, required_action));
        }
        Ok(states)
    }

    fn ranked_actions(&self, env: &CatanEnv, required_action: Option<usize>) -> Vec<usize> {
        let sample = env.clone();
        let mut obs = vec![0.0; OBS_DIM];
        sample.write_obs(&mut obs);
        let mut scratch = NetScratch::new(&self.net);
        self.net.trunk(&obs, &mut scratch);
        let mut valid = Vec::with_capacity(128);
        sample.game().fill_valid_actions(&mut valid);
        let mut ranked: Vec<(f32, usize, Discriminant<Action>)> = valid
            .iter()
            .map(|action| {
                let action_id = encode_action(sample.game(), action);
                (
                    self.net.logit_from(&scratch, action_id),
                    action_id,
                    std::mem::discriminant(action),
                )
            })
            .collect();
        ranked.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(Ordering::Equal));

        // Cover distinct semantic action types before filling remaining roots
        // by policy logit. Without this, large trade/placement families can
        // crowd every counterfactual candidate out of the search.
        let mut best_by_type: HashMap<Discriminant<Action>, (f32, usize)> = HashMap::new();
        for (score, action, action_type) in &ranked {
            best_by_type
                .entry(*action_type)
                .or_insert((*score, *action));
        }
        let mut selected: Vec<(f32, usize)> = best_by_type.into_values().collect();
        selected.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(Ordering::Equal));
        selected.truncate(self.config.root_k);
        let mut selected_ids: Vec<usize> = selected.into_iter().map(|(_, action)| action).collect();
        for (_, action, _) in ranked {
            if selected_ids.len() >= self.config.root_k {
                break;
            }
            if !selected_ids.contains(&action) {
                selected_ids.push(action);
            }
        }
        if let Some(required) = required_action {
            if !selected_ids.contains(&required) {
                if selected_ids.len() >= self.config.root_k {
                    selected_ids.pop();
                }
                selected_ids.push(required);
            }
        }
        selected_ids
    }

    fn greedy_action(&self, env: &mut CatanEnv, scratch: &mut NetScratch) -> usize {
        let mut obs = vec![0.0; OBS_DIM];
        let mut mask = vec![false; NUM_ACTIONS];
        env.write_obs(&mut obs);
        env.write_mask(&mut mask);
        self.net.trunk(&obs, scratch);
        mask.iter()
            .enumerate()
            .filter(|(_, legal)| **legal)
            .max_by(|(left, _), (right, _)| {
                self.net
                    .logit_from(scratch, *left)
                    .partial_cmp(&self.net.logit_from(scratch, *right))
                    .unwrap_or(Ordering::Equal)
            })
            .map(|(action, _)| action)
            .expect("live policy decision must have a legal action")
    }

    fn terminal_score(candidate: usize, winner: i8) -> f32 {
        if winner < 0 {
            0.0
        } else if winner as usize == candidate {
            1.0
        } else {
            -1.0
        }
    }

    fn leaf_score(&self, env: &mut CatanEnv, candidate: usize, scratch: &mut NetScratch) -> f32 {
        let mut obs = vec![0.0; OBS_DIM];
        env.write_obs(&mut obs);
        self.net.trunk(&obs, scratch);
        let value = self.net.value_from(scratch);
        let own_progress = search_potential(env.game(), candidate);
        let best_opponent = (0..env.config.num_players)
            .filter(|seat| *seat != candidate)
            .map(|seat| search_potential(env.game(), seat))
            .fold(0.0f32, f32::max);
        let progress = (own_progress - best_opponent).clamp(-1.0, 1.0);
        let total_weight = self.config.value_weight + self.config.potential_weight;
        (self.config.value_weight * value + self.config.potential_weight * progress) / total_weight
    }

    fn building_points(game: &CatanGame, candidate: usize) -> i32 {
        game.state.settlements_built[candidate] as i32
            + 2 * game.state.cities_built[candidate] as i32
    }

    fn road_position(game: &CatanGame, candidate: usize) -> f32 {
        let award = if game.state.longest_road_player == candidate as i8 {
            1.0
        } else if game.state.longest_road_player >= 0 {
            -1.0
        } else {
            0.0
        };
        let strongest_opponent = (0..game.state.num_players)
            .filter(|seat| *seat != candidate)
            .map(|seat| game.state.road_lengths[seat])
            .max()
            .unwrap_or(0);
        let road_margin =
            (game.state.road_lengths[candidate] as f32 - strongest_opponent as f32) / 15.0;
        award + road_margin
    }

    fn rollout_metrics(
        &self,
        base: &CatanEnv,
        env: &CatanEnv,
        candidate: usize,
        value: f32,
    ) -> RolloutMetrics {
        let target = base.game().state.victory_target.max(1) as f32;
        RolloutMetrics {
            value,
            vp_gain: (env.game().state.calculate_victory_points(candidate)
                - base.game().state.calculate_victory_points(candidate))
                as f32
                / target,
            building_gain: (Self::building_points(env.game(), candidate)
                - Self::building_points(base.game(), candidate)) as f32
                / target,
            road_control: ((Self::road_position(env.game(), candidate)
                - Self::road_position(base.game(), candidate))
                / 2.0)
                .clamp(-1.0, 1.0),
        }
    }

    fn rollout(
        &self,
        base: &CatanEnv,
        candidate: usize,
        root_action: usize,
        seed: u64,
    ) -> RolloutMetrics {
        let mut env = base.search_sample_with_alpha(candidate, seed, self.config.opponent_alpha);
        let mut legal = vec![false; NUM_ACTIONS];
        env.write_mask(&mut legal);
        if !legal[root_action] || !action_executes(&env, root_action) {
            return RolloutMetrics {
                value: -1.0,
                ..RolloutMetrics::default()
            };
        }
        let mut result = env.step(root_action);
        if result.done {
            return self.rollout_metrics(
                base,
                &env,
                candidate,
                Self::terminal_score(candidate, result.winner),
            );
        }
        let mut scratch = NetScratch::new(&self.net);
        for _ in 0..self.config.continuation_decisions {
            let action = self.greedy_action(&mut env, &mut scratch);
            if !action_executes(&env, action) {
                return RolloutMetrics {
                    value: -1.0,
                    ..RolloutMetrics::default()
                };
            }
            result = env.step(action);
            if result.done {
                return self.rollout_metrics(
                    base,
                    &env,
                    candidate,
                    Self::terminal_score(candidate, result.winner),
                );
            }
        }
        let value = self.leaf_score(&mut env, candidate, &mut scratch);
        self.rollout_metrics(base, &env, candidate, value)
    }

    fn action_metrics_inner(
        &self,
        env: &CatanEnv,
        candidate: usize,
        seed: u64,
        required_action: Option<usize>,
    ) -> Vec<(usize, RolloutMetrics)> {
        let actions = self.ranked_actions(env, required_action);
        if actions.len() == 1 {
            return vec![(
                actions[0],
                RolloutMetrics {
                    value: 1.0,
                    ..RolloutMetrics::default()
                },
            )];
        }
        let mut scores: Vec<(usize, RolloutMetrics)> = actions
            .par_iter()
            .enumerate()
            .map(|(action_index, action)| {
                let metrics = (0..self.config.samples)
                    .into_par_iter()
                    .map(|sample| {
                        let rollout_seed =
                            seed ^ (sample as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                        let rollout_seed = if self.config.common_random_numbers {
                            rollout_seed
                        } else {
                            rollout_seed
                                ^ (action_index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        };
                        self.rollout(env, candidate, *action, rollout_seed)
                    })
                    .reduce(RolloutMetrics::default, RolloutMetrics::add)
                    .scale(1.0 / self.config.samples as f32);
                (*action, metrics)
            })
            .collect();
        scores.sort_by(|left, right| {
            right
                .1
                .score(self.config)
                .partial_cmp(&left.1.score(self.config))
                .unwrap_or(Ordering::Equal)
        });
        scores
    }

    fn action_scores_inner(
        &self,
        env: &CatanEnv,
        candidate: usize,
        seed: u64,
        required_action: Option<usize>,
    ) -> Vec<(usize, f32)> {
        self.action_metrics_inner(env, candidate, seed, required_action)
            .into_iter()
            .map(|(action, metrics)| (action, metrics.score(self.config)))
            .collect()
    }

    fn action_inner(&self, env: &CatanEnv, candidate: usize, seed: u64) -> usize {
        self.action_scores_inner(env, candidate, seed, None)
            .first()
            .map(|(action, _)| *action)
            .expect("live policy decision must have a legal action")
    }
}

fn action_executes(env: &CatanEnv, action_id: usize) -> bool {
    let action = decode_action(env.game(), action_id);
    let mut game = env.game().clone();
    game.execute_action(&action)
}

#[cfg(test)]
mod tests {
    use super::{
        coherent_resource_engine, conversion_resource_action, conversion_road_action,
        immediate_vp_action, independent_rollout_seed, pressure_knight_action, refinement_family,
        search_potential, state_action_score, uses_opening_heuristic, uses_opening_road_planning,
        RefinementFamily,
    };
    use catan_core::eval::ValueParams;
    use catan_core::game::{Action, CatanGame, GamePhase, TurnPhase};
    use catan_core::players::HeuristicPlayer;
    use catan_core::state::{DEV_KNIGHT, DEV_VICTORY_POINT};

    #[test]
    fn independent_rollout_seed_separates_candidate_seats() {
        let seat_zero = independent_rollout_seed(11, 23, 0, 4);
        let seat_one = independent_rollout_seed(11, 23, 1, 4);
        assert_ne!(seat_zero, seat_one);
        assert_eq!(seat_zero, independent_rollout_seed(11, 23, 0, 4));
        assert_ne!(seat_zero, independent_rollout_seed(11, 23, 0, 5));
    }

    #[test]
    fn search_potential_rewards_partial_award_progress() {
        let mut game = CatanGame::new_with_target(4, 7, 7);
        game.state.knights_played[0] = 2;
        game.state.road_lengths[0] = 4;
        assert!(search_potential(&game, 0) > search_potential(&game, 1));
    }

    #[test]
    fn opening_hybrid_only_intercepts_setup_settlements() {
        assert!(uses_opening_heuristic(&[
            Action::PlaceInitialSettlement {
                player: 0,
                vertex: 1,
            },
            Action::PlaceInitialSettlement {
                player: 0,
                vertex: 2,
            },
        ]));
        assert!(!uses_opening_heuristic(&[
            Action::PlaceInitialRoad { player: 0, edge: 1 },
            Action::PlaceInitialRoad { player: 0, edge: 2 },
        ]));
        assert!(uses_opening_road_planning(&[
            Action::PlaceInitialRoad { player: 0, edge: 1 },
            Action::PlaceInitialRoad { player: 0, edge: 2 },
        ]));
        assert!(!uses_opening_heuristic(&[Action::BuildRoad {
            player: 0,
            edge: 1,
        }]));
        assert!(!uses_opening_heuristic(&[]));
    }

    #[test]
    fn road_actions_have_a_custom_refinement_family() {
        assert_eq!(
            refinement_family(&Action::MoveRobber { player: 0, tile: 1 }),
            Some(RefinementFamily::Robber)
        );
        assert_eq!(
            refinement_family(&Action::BuildSettlement {
                player: 0,
                vertex: 1,
            }),
            Some(RefinementFamily::Settlement)
        );
        assert_eq!(
            refinement_family(&Action::BuildRoad { player: 0, edge: 1 }),
            Some(RefinementFamily::Road)
        );
    }

    #[test]
    fn coherent_resource_engine_rewards_complete_strategies() {
        assert!(
            coherent_resource_engine([0.3, 0.2, 0.0, 0.0, 0.3])
                > coherent_resource_engine([0.3, 0.0, 0.3, 0.0, 0.2])
        );
        assert_eq!(coherent_resource_engine([0.0; 5]), 0.0);
    }

    #[test]
    fn endgame_conversion_prefers_deterministic_vp_gain() {
        let mut game = CatanGame::new_with_target(4, 1, 7);
        game.game_phase = GamePhase::Playing;
        game.turn_phase = TurnPhase::Main;
        game.state.phase = 1;
        game.state.current_player = 0;
        game.state.settlements_built[0] = 4;
        game.state.dev_cards[0][DEV_VICTORY_POINT] = 2;
        game.state.dev_cards[0][DEV_KNIGHT] = 1;
        game.state.knights_played[0] = 2;
        let actions = [
            Action::PlayKnight { player: 0 },
            Action::EndTurn { player: 0 },
        ];
        assert_eq!(
            immediate_vp_action(&game, 0, &actions, 5, &HeuristicPlayer::v2(1), 0.0, false,),
            Some(Action::PlayKnight { player: 0 })
        );
    }

    #[test]
    fn knight_pressure_plays_until_largest_army_is_safe() {
        let mut game = CatanGame::new_with_target(4, 1, 7);
        game.state.dev_cards[0][DEV_KNIGHT] = 1;
        game.state.knights_played[0] = 3;
        game.state.knights_played[1] = 3;
        game.state.largest_army_player = 0;
        let actions = [
            Action::PlayKnight { player: 0 },
            Action::EndTurn { player: 0 },
        ];
        assert_eq!(
            pressure_knight_action(&game, 0, &actions),
            Some(Action::PlayKnight { player: 0 })
        );
        game.state.knights_played[0] = 5;
        assert_eq!(pressure_knight_action(&game, 0, &actions), None);
    }

    #[test]
    fn endgame_conversion_uses_trade_that_closes_build_deficit() {
        let mut game = CatanGame::new_with_target(4, 1, 7);
        game.game_phase = GamePhase::Playing;
        game.turn_phase = TurnPhase::Main;
        game.state.phase = 1;
        game.state.current_player = 0;
        game.state.settlements_built[0] = 2;
        game.state.dev_cards[0][DEV_VICTORY_POINT] = 3;
        game.state.resources[0] = [0, 4, 0, 0, 3];
        let actions = [
            Action::TradeWithBank {
                player: 0,
                give: 1,
                recv: 0,
            },
            Action::EndTurn { player: 0 },
        ];
        assert_eq!(
            conversion_resource_action(&game, 0, &actions, 5, 5, false),
            Some(Action::TradeWithBank {
                player: 0,
                give: 1,
                recv: 0,
            })
        );
    }

    #[test]
    fn city_conversion_trades_toward_city_instead_of_settlement() {
        let mut game = CatanGame::new_with_target(4, 1, 7);
        game.game_phase = GamePhase::Playing;
        game.turn_phase = TurnPhase::Main;
        game.state.phase = 1;
        game.state.current_player = 0;
        game.state.settlements_built[0] = 2;
        game.state.vertex_road_mask[0] = 1;
        game.state.dev_cards[0][DEV_VICTORY_POINT] = 3;
        game.state.resources[0] = [1, 4, 0, 0, 2];
        let toward_settlement = Action::TradeWithBank {
            player: 0,
            give: 1,
            recv: 2,
        };
        let toward_city = Action::TradeWithBank {
            player: 0,
            give: 1,
            recv: 0,
        };
        let actions = [toward_settlement, toward_city, Action::EndTurn { player: 0 }];
        assert_eq!(
            conversion_resource_action(&game, 0, &actions, 5, 5, true),
            Some(toward_city)
        );
    }

    #[test]
    fn endgame_road_push_ignores_short_roads() {
        let mut game = CatanGame::new_with_target(4, 1, 7);
        game.game_phase = GamePhase::Playing;
        game.turn_phase = TurnPhase::Main;
        game.state.phase = 1;
        game.state.current_player = 0;
        game.state.settlements_built[0] = 2;
        game.state.dev_cards[0][DEV_VICTORY_POINT] = 3;
        let actions = [Action::BuildRoad { player: 0, edge: 0 }];
        assert_eq!(conversion_road_action(&game, 0, &actions, 5), None);
    }

    #[test]
    fn state_refinement_values_city_over_ending_turn() {
        let mut game = CatanGame::new_with_target(4, 1, 7);
        game.game_phase = GamePhase::Playing;
        game.turn_phase = TurnPhase::Main;
        game.state.phase = 1;
        game.state.current_player = 0;
        game.state.settlements_built[0] = 1;
        game.state.vertices[0] = 0;
        game.state.resources[0] = [2, 0, 0, 0, 3];
        let params = ValueParams::default();
        assert!(
            state_action_score(
                &game,
                0,
                &Action::BuildCity {
                    player: 0,
                    vertex: 0,
                },
                &params,
            ) > state_action_score(&game, 0, &Action::EndTurn { player: 0 }, &params)
        );
    }
}

#[pymethods]
impl OpponentAwarePlanner {
    #[new]
    #[pyo3(signature = (
        net_path,
        root_k=4,
        samples=2,
        continuation_decisions=1,
        value_weight=1.0,
        potential_weight=0.25,
        vp_gain_weight=0.0,
        building_gain_weight=0.0,
        road_control_weight=0.0,
        common_random_numbers=false,
        opponent_root_k=1,
        opponent_samples=1,
        opponent_depth=0
    ))]
    fn new(
        net_path: String,
        root_k: usize,
        samples: usize,
        continuation_decisions: usize,
        value_weight: f32,
        potential_weight: f32,
        vp_gain_weight: f32,
        building_gain_weight: f32,
        road_control_weight: f32,
        common_random_numbers: bool,
        opponent_root_k: usize,
        opponent_samples: usize,
        opponent_depth: u32,
    ) -> PyResult<Self> {
        if root_k == 0 || samples == 0 || opponent_root_k == 0 || opponent_samples == 0 {
            return Err(PyValueError::new_err(
                "planner and opponent root_k/samples must be positive",
            ));
        }
        if value_weight < 0.0
            || potential_weight < 0.0
            || vp_gain_weight < 0.0
            || building_gain_weight < 0.0
            || road_control_weight < 0.0
        {
            return Err(PyValueError::new_err(
                "planner score weights must be non-negative",
            ));
        }
        if value_weight
            + potential_weight
            + vp_gain_weight
            + building_gain_weight
            + road_control_weight
            <= 0.0
        {
            return Err(PyValueError::new_err(
                "at least one planner score weight must be positive",
            ));
        }
        Ok(Self {
            net: Arc::new(MlpNet::load(net_path.as_ref())),
            config: PlannerConfig {
                root_k,
                samples,
                continuation_decisions,
                value_weight,
                potential_weight,
                vp_gain_weight,
                building_gain_weight,
                road_control_weight,
                common_random_numbers,
                opponent_alpha: AlphaConfig {
                    root_k: opponent_root_k,
                    samples: opponent_samples,
                    depth: opponent_depth,
                },
            },
        })
    }

    fn action(&self, py: Python<'_>, env: PyRef<'_, Env>, seed: u64) -> PyResult<usize> {
        let candidate = Self::validate_env(&env.inner)?;
        let base = env.inner.clone();
        Ok(py.allow_threads(|| self.action_inner(&base, candidate, seed)))
    }

    /// Counterfactual root scores for selected live vectorized environments.
    ///
    /// Each result contains `(encoded_action, mean_rollout_score)` pairs in
    /// descending score order. `required_actions` is normally the deployed
    /// action and guarantees that it is evaluated alongside the alternatives.
    /// The cloned environments retain their configured opponents.
    fn score_indices(
        &self,
        py: Python<'_>,
        env: PyRef<'_, VecEnv>,
        indices: Vec<usize>,
        seeds: Vec<u64>,
        required_actions: Vec<usize>,
    ) -> PyResult<Vec<Vec<(usize, f32)>>> {
        let states = Self::scoring_states(&env, indices, seeds, required_actions)?;
        Ok(py.allow_threads(|| {
            states
                .into_par_iter()
                .map(|(base, candidate, seed, required_action)| {
                    self.action_scores_inner(&base, candidate, seed, Some(required_action))
                })
                .collect()
        }))
    }

    /// Counterfactual action metrics for conversion-aware policy targets.
    ///
    /// Rows contain `(action, composite_score, value, vp_gain,
    /// building_gain, road_control)` and are sorted by composite score.
    fn score_conversion_indices(
        &self,
        py: Python<'_>,
        env: PyRef<'_, VecEnv>,
        indices: Vec<usize>,
        seeds: Vec<u64>,
        required_actions: Vec<usize>,
    ) -> PyResult<Vec<Vec<ActionMetricRow>>> {
        let states = Self::scoring_states(&env, indices, seeds, required_actions)?;
        Ok(py.allow_threads(|| {
            states
                .into_par_iter()
                .map(|(base, candidate, seed, required_action)| {
                    self.action_metrics_inner(&base, candidate, seed, Some(required_action))
                        .into_iter()
                        .map(|(action, metrics)| {
                            (
                                action,
                                metrics.score(self.config),
                                metrics.value,
                                metrics.vp_gain,
                                metrics.building_gain,
                                metrics.road_control,
                            )
                        })
                        .collect()
                })
                .collect()
        }))
    }
}
