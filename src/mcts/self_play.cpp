#include "self_play.hpp"
#include "./mcts_model.hpp"
#include "./node.hpp"
#include <iostream>

#include <vector>


void SelfPlayWorker::grind(){
    while(true) {
        play_game();
    }

}

void SelfPlayWorker::play_game(){
    mcts_model::Model mcts(chess::side_white);
    chess::position state;

    std::vector<GameRow> game_rows;
    size_t moves = 0;


    while(!state.is_checkmate() && !state.is_stalemate() && moves++ < config::MAX_MOVES) {
        node::Node root = mcts.search(state, config::max_iter);
        game_rows.emplace_back(state, root.action_distribution(num_actions));
        chess::move move = root->best_move();
        state.make_move(move);
    }
    std::cout << "Self play game done. Rows: " << game_rows.size() << std::endl;
}
