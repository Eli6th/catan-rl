//! Hidden-information helpers for search and belief supervision.

use catan_core::game::CatanGame;
use catan_core::state::{NUM_DEV_CARD_TYPES, NUM_RESOURCES};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Re-sample information hidden from `observer` while preserving all public
/// counts and the observer's exact private cards.
///
/// Resource identities are sampled from the publicly derivable pool
/// (supply minus bank minus observer hand). Development cards are shuffled
/// across opponent hands and the remaining deck while preserving every
/// public hand/deck count. Future chance outcomes use a fresh RNG stream.
pub fn redeterminize(game: &mut CatanGame, observer: usize, seed: u64) {
    let state = &mut game.state;
    assert!(observer < state.num_players);
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut resource_pool = Vec::with_capacity(64);
    for r in 0..NUM_RESOURCES {
        for p in 0..state.num_players {
            if p != observer {
                for _ in 0..state.resources[p][r] {
                    resource_pool.push(r as i16);
                }
            }
        }
    }
    resource_pool.shuffle(&mut rng);
    let resource_counts: Vec<usize> = (0..state.num_players)
        .map(|p| state.total_resources(p) as usize)
        .collect();
    for p in 0..state.num_players {
        if p != observer {
            state.resources[p] = [0; NUM_RESOURCES];
        }
    }
    let mut offset = 0;
    for (p, &count) in resource_counts.iter().enumerate().take(state.num_players) {
        if p == observer {
            continue;
        }
        for &resource in &resource_pool[offset..offset + count] {
            state.resources[p][resource as usize] += 1;
        }
        offset += count;
    }
    debug_assert_eq!(offset, resource_pool.len());

    let dev_counts: Vec<usize> = (0..state.num_players)
        .map(|p| state.dev_cards[p].iter().map(|&n| n as usize).sum())
        .collect();
    let bought_count: usize = state
        .dev_cards_bought_this_turn
        .iter()
        .map(|&n| n as usize)
        .sum();
    let turn_owner = state.current_player;
    let mut dev_pool = Vec::with_capacity(32);
    for p in 0..state.num_players {
        if p == observer {
            continue;
        }
        for card in 0..NUM_DEV_CARD_TYPES {
            for _ in 0..state.dev_cards[p][card] {
                dev_pool.push(card as i8);
            }
        }
        state.dev_cards[p] = [0; NUM_DEV_CARD_TYPES];
    }
    dev_pool.extend_from_slice(&state.dev_deck[state.dev_deck_idx..]);
    dev_pool.shuffle(&mut rng);

    offset = 0;
    for (p, &count) in dev_counts.iter().enumerate().take(state.num_players) {
        if p == observer {
            continue;
        }
        for &card in &dev_pool[offset..offset + count] {
            state.dev_cards[p][card as usize] += 1;
        }
        offset += count;
    }
    let remaining = state.dev_deck.len() - state.dev_deck_idx;
    state.dev_deck[state.dev_deck_idx..].copy_from_slice(&dev_pool[offset..offset + remaining]);

    if turn_owner != observer {
        state.dev_cards_bought_this_turn = [0; NUM_DEV_CARD_TYPES];
        let mut held = Vec::with_capacity(dev_counts[turn_owner]);
        for card in 0..NUM_DEV_CARD_TYPES {
            for _ in 0..state.dev_cards[turn_owner][card] {
                held.push(card);
            }
        }
        held.shuffle(&mut rng);
        for &card in held.iter().take(bought_count) {
            state.dev_cards_bought_this_turn[card] += 1;
        }
    }

    state.rng = SmallRng::seed_from_u64(seed ^ 0xA5A5_5A5A_D3C4_B2E1);
}

/// Exact hidden-card labels, seat-relative to `observer`, matching the
/// observation encoder's three opponent-private blocks.
pub fn private_targets(game: &CatanGame, observer: usize, out: &mut [f32]) {
    assert_eq!(out.len(), 30);
    out.fill(0.0);
    let state = &game.state;
    for rel in 1..state.num_players {
        let p = (observer + rel) % state.num_players;
        let base = (rel - 1) * 10;
        for r in 0..NUM_RESOURCES {
            out[base + r] = state.resources[p][r] as f32 / 19.0;
            out[base + 5 + r] = state.dev_cards[p][r] as f32 / 5.0;
        }
    }
}
