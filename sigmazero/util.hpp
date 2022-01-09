#ifndef UTIL_H
#define UTIL_H

#include <random>
#include <chess/chess.hpp>

#include "drl/sigmanet.hpp"


/**
 * Returns the global pseudo-random number generator
 */
std::mt19937& get_generator();

chess::game get_simple_game();

void debug_position(sigmanet& model, torch::Device device, chess::position state);

#endif // UTIL_H