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

    void grind();
    void play_game();

};


#endif // SELF_PLAY_H