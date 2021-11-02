#include "self_play.hpp"
#include "mcts.hpp"

#include "node.hpp"
#include "network.hpp"
#include <iostream>
#include <memory>

#include <vector>


void SelfPlayWorker::grind(){
    while(true) {
        play_game();
    }

}

void SelfPlayWorker::play_game(){
    chess::position state = chess::position::from_fen(chess::position::fen_start);
    mcts::Network network;
    std::vector<SelfPlayWorker::GameRow> game_rows;
    size_t moves = 0;


    while(!state.is_checkmate() && !state.is_stalemate() && moves++ < max_moves) {
        std::shared_ptr<mcts::Node> root = mcts::mcts(state, max_iter, network);
        GameRow row = {state, root->action_distribution(num_actions)};
        game_rows.push_back(row);
        chess::move move = root->best_move();
        state.make_move(move);
    }
    std::cout << "Self play game done. Rows: " << game_rows.size() << std::endl;
}
