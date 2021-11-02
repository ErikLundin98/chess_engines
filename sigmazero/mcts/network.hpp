#ifndef NETWORK_H
#define NETWORK_H

#include <chess/chess.hpp>
#include <unordered_map>
#include <algorithm>

struct Network {
    struct Evaluation {
        double value;
        std::unordered_map<size_t, double> action_probabilities;
    };
    
    Evaluation evaluate(chess::position& state);

    // SHOULD EXIST SIMPLER WAY, DO LATER
    static size_t action_from_move(chess::position state, chess::move& move);


    static chess::move move_from_action(size_t action);

};


#endif // NETWORK_H