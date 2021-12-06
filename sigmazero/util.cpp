#include "util.hpp"
#include <string>
#include <vector>
#include "drl/action_encodings.hpp"
#include <utility>
#include <algorithm>
#include <iostream>

std::mt19937& get_generator() {
    static std::mt19937 generator = std::mt19937(std::random_device{}());
    return generator;
}

chess::game get_simple_game (){
    static std::vector<std::string> start_positions = {
        "8/7k/R7/8/8/8/1B4R1/6K1 w - - 0 1", // Mate in two, rook lane cornering
        "8/1b4rk/8/8/8/r7/7K/8 b - - 0 1", // BLACK
        "8/7k/R7/8/8/8/6R1/6K1 w - - 0 1", // Same as above but no bishop, so mate in three
        "8/5P1k/R7/8/8/8/1B4R1/6K1 w - - 0 1" // Cheeky mate in one with underpromotion to knight
    };
    chess::position p = chess::position::from_fen(start_positions[0]);
    chess::game game{p, std::vector<chess::move>()};
    return game;

}

void debug_position(sigmanet& model, torch::Device device, chess::position state){
    auto [value, policy] = model->evaluate(state, device);
    std::vector<std::pair<double, size_t>> prob_actions;
    for (auto [action, prob]: policy) {
        prob_actions.push_back(std::make_pair(prob, action));
    }
    std::sort(prob_actions.begin(), prob_actions.end());
    std::reverse(prob_actions.begin(), prob_actions.end());
    std::setprecision(4);
    std::cerr << "state("<< value << "): " << state.to_fen() << std::endl;
    for (auto [prob, action]: prob_actions) {
        chess::move move = action_encodings::move_from_action(state, action);
        chess::undo undo = state.make_move(move);
        auto [value2, _] = model->evaluate(state, device);
        state.undo_move(move, undo);
        std::setprecision(4);
        std::cerr << move.to_lan() << ":" << prob << '(' << value2 <<  ")  |  ";
    }
    std::cerr << std::endl;
}