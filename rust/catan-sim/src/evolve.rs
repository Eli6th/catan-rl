//! Genetic algorithm over the heuristic's weights: fitness = win rate vs
//! three frozen Heuristic-v1 opponents. Every individual in a generation
//! faces the SAME game seeds (common random numbers), so fitness gaps
//! reflect skill, not board luck. Output: a Heuristic-v2 candidate.

use catan_core::eval::{GreedyValuePlayer, ValueParams};
use catan_core::game::CatanGame;
use catan_core::players::{play_game, HeuristicParams, HeuristicPlayer, Player};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const POP: usize = 20;
const ELITE: usize = 4;
const GENS: usize = 12;

fn fitness(params: HeuristicParams, games: u64, seed_base: u64) -> f32 {
    let wins: u32 = (0..games)
        .map(|g| {
            let game_seed = seed_base.wrapping_add(g.wrapping_mul(0x9E3779B97F4A7C15));
            // Rotate the candidate's seat so seat advantage washes out.
            let my_seat = (g % 4) as usize;
            let mut game = CatanGame::new_with_target(4, game_seed, 7);
            game.record_history = false;
            let mut players: Vec<Box<dyn Player>> = (0..4)
                .map(|s| -> Box<dyn Player> {
                    if s == my_seat {
                        Box::new(HeuristicPlayer::with_params(game_seed ^ 1, params))
                    } else {
                        Box::new(HeuristicPlayer::new(game_seed ^ (s as u64 + 2)))
                    }
                })
                .collect();
            u32::from(play_game(&mut game, &mut players) == my_seat as i8)
        })
        .sum();
    wins as f32 / games as f32
}

fn mutate(params: HeuristicParams, rng: &mut SmallRng, sigma: f32) -> HeuristicParams {
    let defaults = HeuristicParams::default().to_array();
    let mut a = params.to_array();
    for (i, v) in a.iter_mut().enumerate() {
        // Gaussian-ish noise scaled to each gene's natural magnitude.
        let scale = defaults[i].abs().max(0.05);
        let noise: f32 = (0..4).map(|_| rng.gen::<f32>()).sum::<f32>() / 2.0 - 1.0;
        *v = (*v + noise * sigma * scale).max(0.0);
    }
    HeuristicParams::from_array(a)
}

fn crossover(a: HeuristicParams, b: HeuristicParams, rng: &mut SmallRng) -> HeuristicParams {
    let (aa, ba) = (a.to_array(), b.to_array());
    let mut child = [0f32; 6];
    for i in 0..6 {
        child[i] = if rng.gen::<bool>() { aa[i] } else { ba[i] };
    }
    HeuristicParams::from_array(child)
}

pub fn evolve(games_per_eval: u64, seed: u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    // Individual 0 is v1 itself: evolution must BEAT the incumbent to count.
    let mut population: Vec<HeuristicParams> = vec![HeuristicParams::default()];
    while population.len() < POP {
        let base = HeuristicParams::default();
        population.push(mutate(base, &mut rng, 0.6));
    }

    for gen in 0..GENS {
        let seed_base = seed + gen as u64 * 1_000_003; // CRN per generation
        let mut scored: Vec<(f32, HeuristicParams)> = population
            .par_iter()
            .map(|&p| (fitness(p, games_per_eval, seed_base), p))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let best = scored[0];
        let mean: f32 = scored.iter().map(|s| s.0).sum::<f32>() / scored.len() as f32;
        println!(
            "gen {gen:2}: best {:.1}% | mean {:.1}% | best params {:?}",
            best.0 * 100.0,
            mean * 100.0,
            best.1.to_array().map(|v| (v * 100.0).round() / 100.0),
        );

        let sigma = 0.4 * (1.0 - gen as f32 / GENS as f32) + 0.08;
        let elites: Vec<HeuristicParams> = scored.iter().take(ELITE).map(|s| s.1).collect();
        population = elites.clone();
        while population.len() < POP {
            let a = elites[rng.gen_range(0..ELITE)];
            let b = scored[rng.gen_range(0..POP / 2)].1;
            population.push(mutate(crossover(a, b, &mut rng), &mut rng, sigma));
        }
    }

    // Holdout: fresh seeds, larger sample, champion vs the v1 incumbent.
    let champion = {
        let seed_base = seed + 777_777;
        let mut scored: Vec<(f32, HeuristicParams)> = population
            .par_iter()
            .map(|&p| (fitness(p, games_per_eval, seed_base), p))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored[0].1
    };
    let holdout_champ = fitness(champion, games_per_eval * 6, seed + 31_337);
    let holdout_v1 = fitness(
        HeuristicParams::default(),
        games_per_eval * 6,
        seed + 31_337,
    );
    println!(
        "\nholdout ({} fresh games, seat-rotated, vs 3x Heuristic-v1):",
        games_per_eval * 6
    );
    println!(
        "  v1 baseline : {:.1}% (fair share = 25%)",
        holdout_v1 * 100.0
    );
    println!("  evolved     : {:.1}%", holdout_champ * 100.0);
    println!("  params      : {:?}", champion);
    let json = format!(
        "{{\"diversity_bonus\":{},\"port_bonus\":{},\"robber_steal_bonus\":{},\
         \"robber_self_penalty\":{},\"steal_cards_weight\":{},\"trade_accept_threshold\":{},\
         \"holdout_win_rate\":{}}}",
        champion.diversity_bonus,
        champion.port_bonus,
        champion.robber_steal_bonus,
        champion.robber_self_penalty,
        champion.steal_cards_weight,
        champion.trade_accept_threshold,
        holdout_champ,
    );
    std::fs::write("heuristic_v2_candidate.json", &json).ok();
    println!("written to heuristic_v2_candidate.json");
}

const VALUE_POP: usize = 32;
const VALUE_ELITE: usize = 6;
const VALUE_GENS: usize = 24;

fn value_fitness(params: ValueParams, games: u64, seed_base: u64) -> f32 {
    let wins: u32 = (0..games)
        .into_par_iter()
        .map(|g| {
            let game_seed = seed_base.wrapping_add(g.wrapping_mul(0x9E3779B97F4A7C15));
            let my_seat = (g % 4) as usize;
            let mut game = CatanGame::new_with_target(4, game_seed, 7);
            game.record_history = false;
            let mut players: Vec<Box<dyn Player>> = (0..4)
                .map(|seat| -> Box<dyn Player> {
                    if seat == my_seat {
                        Box::new(GreedyValuePlayer::with_params(game_seed ^ 1, params))
                    } else {
                        Box::new(HeuristicPlayer::v2(game_seed ^ (seat as u64 + 2)))
                    }
                })
                .collect();
            u32::from(play_game(&mut game, &mut players) == my_seat as i8)
        })
        .sum();
    wins as f32 / games as f32
}

fn mutate_value(params: ValueParams, rng: &mut SmallRng, sigma: f32) -> ValueParams {
    let defaults = ValueParams::default().to_array();
    let mut values = params.to_array();
    for (index, value) in values.iter_mut().enumerate() {
        let scale = defaults[index].abs().max(0.05);
        let noise = (0..4).map(|_| rng.gen::<f32>()).sum::<f32>() / 2.0 - 1.0;
        *value = (*value + noise * sigma * scale).max(0.0);
    }
    ValueParams::from_array(values)
}

fn crossover_value(a: ValueParams, b: ValueParams, rng: &mut SmallRng) -> ValueParams {
    let (a, b) = (a.to_array(), b.to_array());
    let mut child = [0.0f32; catan_core::eval::NUM_VALUE_PARAMS];
    for index in 0..child.len() {
        child[index] = if rng.gen::<bool>() {
            a[index]
        } else {
            b[index]
        };
    }
    ValueParams::from_array(child)
}

pub fn evolve_value(games_per_eval: u64, seed: u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut population = vec![ValueParams::default()];
    while population.len() < VALUE_POP {
        population.push(mutate_value(ValueParams::default(), &mut rng, 0.7));
    }

    for generation in 0..VALUE_GENS {
        let seed_base = seed + generation as u64 * 1_000_003;
        let mut scored: Vec<(f32, ValueParams)> = population
            .par_iter()
            .map(|&params| (value_fitness(params, games_per_eval, seed_base), params))
            .collect();
        scored.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap());
        let best = scored[0];
        let mean = scored.iter().map(|entry| entry.0).sum::<f32>() / scored.len() as f32;
        println!(
            "value gen {generation:2}: best {:.1}% | mean {:.1}% | params {:?}",
            best.0 * 100.0,
            mean * 100.0,
            best.1.to_array(),
        );

        let sigma = 0.45 * (1.0 - generation as f32 / VALUE_GENS as f32) + 0.05;
        let elites: Vec<ValueParams> = scored
            .iter()
            .take(VALUE_ELITE)
            .map(|entry| entry.1)
            .collect();
        population = elites.clone();
        while population.len() < VALUE_POP {
            let a = elites[rng.gen_range(0..VALUE_ELITE)];
            let b = scored[rng.gen_range(0..VALUE_POP / 2)].1;
            population.push(mutate_value(
                crossover_value(a, b, &mut rng),
                &mut rng,
                sigma,
            ));
        }
    }

    let mut finalists: Vec<(f32, ValueParams)> = population
        .par_iter()
        .map(|&params| {
            (
                value_fitness(params, games_per_eval * 4, seed + 777_777),
                params,
            )
        })
        .collect();
    finalists.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap());
    let champion = finalists[0].1;
    let holdout_games = (games_per_eval * 16).max(4096);
    let holdout = value_fitness(champion, holdout_games, seed + 31_337);
    let baseline = value_fitness(ValueParams::default(), holdout_games, seed + 31_337);
    println!(
        "\nvalue holdout ({holdout_games} fresh games vs 3x Heuristic-v2):\n  baseline: {:.1}%\n  evolved : {:.1}%\n  params  : {:?}",
        baseline * 100.0,
        holdout * 100.0,
        champion.to_array(),
    );
    let json = serde_json::json!({
        "params": champion.to_array(),
        "holdout_games": holdout_games,
        "baseline_win_rate": baseline,
        "holdout_win_rate": holdout,
    });
    std::fs::write(
        "value_params_candidate.json",
        serde_json::to_vec_pretty(&json).unwrap(),
    )
    .expect("write value_params_candidate.json");
}
