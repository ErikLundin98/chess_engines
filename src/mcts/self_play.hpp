#ifndef SELF_PLAY_H
#define SELF_PLAY_H

#include "./mcts_model.hpp"
#include "./node.hpp"

#include <vector>


size_t num_actions = 64*73;
size_t max_iter = 800;
size_t MAX_MOVES = 500;

struct SelfPlayWorker{

    struct GameRow {
        chess::position state;
        std::vector<double> action_distribution;
    };

    void grind(){
        while(true) {
            play_game();
        }

    }

    void play_game(){
        mcts_model::TimedModel mcts;
        chess::position state;

        std::vector<GameRow> game_rows;
        size_t moves = 0;


        while(!state.is_checkmate() && !state.is_stalemate() && moves++ < MAX_MOVES) {
            node::Node root = mcts.search(state, max_iter);
            game_rows.emplace_back(state, root.action_distribution(num_actions))
            chess::move move = root->best_move();
            state.make_move(move);
        }
    }

};


#endif // SELF_PLAY_H