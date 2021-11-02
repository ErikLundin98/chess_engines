#include "mcts.hpp"
#include "node.hpp"
#include "misc.hpp"
#include "network.hpp"
#include <chess/chess.hpp>
#include <memory>

namespace mcts
{

    std::shared_ptr<Node> mcts(chess::position state, int max_iter, const mcts::Network& network)
    {
        auto initialize_node = [&network](std::shared_ptr<Node> node, chess::position state){
            Network::Evaluation result = network.evaluate(state);
            node->expand(result.action_probabilities);
            node->backpropagate(result.value);
        };
        std::shared_ptr<Node> main_node{std::make_shared<Node>(state)};
        initialize_node(main_node, state);
        main_node->add_exploration_noise(0.3, 0.25);
        for(int i = 0 ; i < max_iter ; ++i)
        {
            std::shared_ptr<Node> current_node = main_node->traverse();
            if(!current_node->is_over()) {
                current_node->backpropagate(current_node->get_terminal_value());
                continue;
            }
            initialize_node(current_node, state);
        }
        return main_node;
    }
}
