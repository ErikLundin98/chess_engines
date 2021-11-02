#include "./mcts_model.hpp"
#include <mcts/node.hpp>
#include <mcts/misc.hpp>
#include <mcts/rollout.hpp>
#include <mcts/policy.hpp>
#include <chess/chess.hpp>
#include <memory>
#include <string>

namespace mcts_model
{
    
    Model::Model(chess::side model_side) 
        : model_side{model_side}
    {}

    void initialize_node(Node& node){
        Network network;
        Network::Evaluation result = network.evaluate();
        node->expand(result);
        node->initialize_value(result.value);
    }


    node::Node Model::search(chess::position state, int max_iter)
    {
        
        std::shared_ptr<node::Node> main_node{std::make_shared<node::Node>(state, model_side)};
        initialize_node(main_node);
        main_node->add_exploration_noise(0.3, 0.25);
        for(int i = 0 ; i < max_iter ; ++i)
        {
            std::shared_ptr<node::Node> current_node = main_node->traverse();
            // What to do (expand, evaluate) if:
            // * Current Node is terminal
            // * Current node has n=0
            if(current_node->is_over()) break;

            initialize_node(current_node);
            current_node->backpropagate();
        }
        return main_node;
    }
    // TimedModel::TimedModel(rollout_type rollout, policy_type policy, chess::side model_side) 
    // : Model{rollout, policy, model_side}
    // {}

    // chess::move TimedModel::search(chess::position state, int max_iter)
    // {
    //     outer_timer.set_start();

    //     std::shared_ptr<node::Node> main_node{std::make_shared<node::Node>(state, model_side)};
    //     inner_timer.set_start();
    //     main_node->expand();
    //     t_expanding += inner_timer.get_time();
    //     for(int i = 0 ; i < max_iter ; ++i)
    //     {
    //         inner_timer.set_start();
    //         std::shared_ptr<node::Node> current_node = main_node->traverse();
    //         t_traversing += inner_timer.get_time();
    //         if(current_node->is_over()) break;
    //         if(current_node->get_n() != 0)
    //         {
    //             inner_timer.set_start();
    //             current_node->expand();
    //             t_expanding += inner_timer.get_time();
    //             current_node = current_node->get_children().front();
    //         }
    //         inner_timer.set_start();
    //         current_node->rollout(rollout, policy);
    //         t_rollouting += inner_timer.get_time(true);
    //         current_node->backpropagate();
    //         t_backpropping += inner_timer.get_time();

            
    //     }
    //     t_tot += outer_timer.get_time();
        
    //     return main_node->best_move();
    // }


    // std::string TimedModel::time_report() 
    // {
    //     std::string report = "-- Time Report --";
    //     report += "\nexpanding: " + std::to_string(t_expanding);
    //     report += "\ntraversing: " + std::to_string(t_traversing);
    //     report += "\nrolling out: " + std::to_string(t_rollouting);
    //     report += "\nbackpropagating: " + std::to_string(t_backpropping);
    //     report += "\neverything else: " + std::to_string(t_tot-t_expanding-t_traversing-t_rollouting-t_backpropping);
    //     return report;
    // }
    
}
