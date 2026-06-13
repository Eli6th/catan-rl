use catan_core::game::{Action, CatanGame};
use catan_env::redeterminize;

fn advance_setup(game: &mut CatanGame) {
    let mut valid = Vec::new();
    while game.state.phase == 0 {
        game.fill_valid_actions(&mut valid);
        assert!(!valid.is_empty());
        game.execute_action(&valid[0]);
    }
}

#[test]
fn redeterminization_preserves_public_counts_and_observer_cards() {
    let mut game = CatanGame::new(4, 42);
    advance_setup(&mut game);
    for p in 0..4 {
        game.state.resources[p] = [p as i16 + 1, 2, 1, 0, 1];
        game.state.dev_cards[p] = [p as i8, 1, 0, 0, 0];
    }
    game.state.bank = [9, 11, 15, 19, 15];
    game.state.dev_deck_idx = 8;
    let observer = game.current_player();
    let own_resources = game.state.resources[observer];
    let own_dev = game.state.dev_cards[observer];
    let totals: Vec<i16> = (0..4).map(|p| game.state.total_resources(p)).collect();
    let dev_totals: Vec<i8> = (0..4)
        .map(|p| game.state.dev_cards[p].iter().sum())
        .collect();

    redeterminize(&mut game, observer, 99);

    assert_eq!(game.state.resources[observer], own_resources);
    assert_eq!(game.state.dev_cards[observer], own_dev);
    for p in 0..4 {
        assert_eq!(game.state.total_resources(p), totals[p]);
        assert_eq!(game.state.dev_cards[p].iter().sum::<i8>(), dev_totals[p]);
    }
}

#[test]
fn redeterminized_games_remain_playable() {
    let mut game = CatanGame::new(4, 7);
    let mut valid = Vec::new();
    for step in 0..500 {
        if game.is_game_over() {
            break;
        }
        let observer = game.current_player();
        let mut sampled = game.clone();
        redeterminize(&mut sampled, observer, step + 100);
        sampled.fill_valid_actions(&mut valid);
        assert!(!valid.is_empty());
        let action: Action = valid[0];
        assert!(sampled.execute_action(&action));

        game.fill_valid_actions(&mut valid);
        assert!(!valid.is_empty());
        assert!(game.execute_action(&valid[0]));
    }
}
