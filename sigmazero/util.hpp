#include <random>
#include <chess/chess.hpp>
/**
 * Returns the global pseudo-random number generator
 */
std::mt19937& get_generator();

chess::game get_simple_game();