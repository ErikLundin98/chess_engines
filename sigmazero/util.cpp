#include "util.hpp"
#include <string>
#include <vector>


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