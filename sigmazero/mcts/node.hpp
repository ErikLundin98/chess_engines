#ifndef NODE_H
#define NODE_H

#include "network.hpp"
#include <chess/chess.hpp>
#include "misc.hpp"
#include <vector>
#include <float.h>
#include <memory>

namespace mcts
{    
class Node : public std::enable_shared_from_this<Node>
{

    public:
        // Used to create a node that is not a parent node
        Node(chess::position state, bool is_start_node, std::weak_ptr<Node> parent, chess::move move);

        // Used to create a parent node
        Node(chess::position state);
        ~Node() = default;
        // Get child nodes
        inline std::vector<std::shared_ptr<Node>> get_children() const
        {
            return this->children;
        }


        // Backpropagate score and visits to parent node
        void backpropagate(double value);

        // Expand node
        void expand(const std::unordered_map<size_t, double>& action_probabilities);


        void explore_and_set_priors(const Network &network);

        void add_exploration_noise(double dirichlet_alpha, double exploration_factor);

        double get_value() const;

        double get_terminal_value() const;


        // UCB1 scoring function
        inline double UCB1() const
        {
            auto p = parent.lock();
            if (n == 0 || (p && p->n==0))
            {
                return DBL_MAX;
            }
            else
            {
                int N = p ? p->n : 1;

                double explore_factor = log((N + pb_c_base + 1)/pb_c_base) + pb_c_init;
                explore_factor *= sqrt(N) / (n + 1);

                double prior_score = explore_factor * prior;
                double value_score = -get_value();
                // Negative value score because UCB is useful from the perspective
                // of the parent. The parent want's the child to be in a bad position, because the child
                // is the opponents turn.
                return prior_score + value_score; 
            }
        }

        // Determine next node to expand/rollout by traversing tree
        std::shared_ptr<Node> traverse();

        // Retrieve the best child node based on UCB1 score
        // Can be useful if we want to keep the tree from the previous iterations
        std::shared_ptr<Node> best_child() const;

        // Get action distribution for the children of this node.
        // Should be ran after the entire mcts search is completeted.
        std::vector<double> action_distribution(size_t num_actions);

        // Get the move that gives the best child
        // Useful for baseline mcts algorithm
        chess::move best_move() const;

        // Get state
        chess::position get_state() const;
        // Check if current state is a terminal state
        bool is_over() const;
        // Get amount of vists
        int get_n() const;

        // Print the main node and its children
        std::string to_string(int layers_left=1) const;

        static double WIN_SCORE;
        static double DRAW_SCORE;
        static double pb_c_base;
        static double pb_c_init;

    private: // Bad, but hate private stuff
        
        chess::position state;
        chess::move move;
        bool is_start_node;
        bool is_terminal_node = false;
        std::weak_ptr<Node> parent;
        std::vector<std::shared_ptr<Node>> children;
        double t = 0.0;
        int n = 0;
        double prior;
        size_t action;
};

// Initialize node library 
// Sets reward scores
// Not necessary unless modifying scores is desired

} //namespace node

#endif /* NODE_H */