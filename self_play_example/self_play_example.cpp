#include <chess/chess.hpp>
#include <mcts/node.hpp>
#include <mcts/self_play.hpp>
#include <iostream>
#include <string>
#include <memory>
#include <unordered_map>

// Usage: ./main MCTS_ITER <config filename>

int main(int argc, char* argv[])
{
    chess::init();
    node::init(WIN_SCORE, DRAW_SCORE, 2.0);
    SelfPlayWorker worker;
    worker.grind();
    return 0;
}
