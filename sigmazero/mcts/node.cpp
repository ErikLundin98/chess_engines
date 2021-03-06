
#include "misc.hpp"
#include "node.hpp"
#include <chess/chess.hpp>
#include <sigmazero/drl/action_encodings.hpp>
#include <sigmazero/util.hpp>
#include <random>
#include <vector>
#include <iostream>
#include <memory>
#include <unordered_map>

namespace mcts
{
    // Used to create a node that is not a parent node
    Node::Node(chess::position state, bool is_start_node, std::weak_ptr<Node> parent, chess::move move)
        : state{state},
          is_start_node{is_start_node},
          parent{parent},
          move{move}
    {
        this->children = {};
        this->t = 0;
        this->n = 0;
    }
    // Used to create a parent node
    Node::Node(chess::position state) : Node(state, true, std::weak_ptr<Node>(), chess::move()) {}

    std::vector<std::shared_ptr<Node>> Node::get_children() const
    {
        return this->children;
    }

    // Backpropagate score and visits to parent node
    void Node::backpropagate(double value)
    {
        t += value;
        n++;
        if (is_start_node)
        {
            return;
        }
        if (auto p = parent.lock())
        {
            p->backpropagate(-value);
        }
    }

    double Node::get_terminal_value() const
    {
        return state.is_checkmate() ? -WIN_SCORE : DRAW_SCORE;
    }

    double Node::UCB1() const
    {
        auto p = parent.lock();
        // if (n == 0 || (p && p->n == 0))
        // {
        //     return DBL_MAX;
        // }
        // else
        // {
        int N = p ? p->n : 1;

        double explore_factor = log((N + pb_c_base + 1) / pb_c_base) + pb_c_init;
        explore_factor *= sqrt(N) / (n + 1);

        double prior_score = explore_factor * prior;
        double value_score = -get_value();
        // Negative value score because UCB is useful from the perspective
        // of the parent. The parent wants the child to be in a bad position, because the child
        // is the opponents turn.
        return prior_score + value_score;
        // }
    }

    // Expand node
    void Node::expand(const std::unordered_map<size_t, double> &action_probabilities)
    {

        for (auto [child_action, child_prior] : action_probabilities)
        {
            // Actions will be from "white perspective" (even if black)
            // Therefor, to get actual libchess move we first have to conditionally
            // flip the action (the condition being if we are black)
            size_t libchess_action = action_encodings::cond_flip_action(state, child_action);
            chess::move child_move = action_encodings::move_from_action(state, libchess_action);
            chess::position child_state = state.copy_move(child_move);
            std::shared_ptr<Node> new_child = std::make_shared<Node>(child_state, false, weak_from_this(), child_move);
            new_child->prior = child_prior;
            new_child->action = child_action;
            children.push_back(new_child);
        }
    }

    void Node::explore_and_set_priors(const std::pair<double, std::unordered_map<size_t, double>>& evaluation)
    {
        if(children.size() == 0) { // only expand if not already expanded
            expand(evaluation.second);
        }
        backpropagate(evaluation.first);
    }

    void Node::add_exploration_noise(double dirichlet_alpha, double exploration_factor)
    {
        std::gamma_distribution<double> gamma_distribution(dirichlet_alpha, 1.0);
        for (std::shared_ptr<Node> child : children)
        {
            double noise = gamma_distribution(get_generator());
            child->prior = child->prior * (1 - exploration_factor) + exploration_factor * noise;
        }
    }

    // Determine next node to expand/rollout by traversing tree
    std::shared_ptr<Node> Node::traverse()
    {
        std::vector<double> UCB1_scores{};
        for (std::shared_ptr<Node> child : children)
        {
            UCB1_scores.push_back(child->UCB1());
        }
        
        std::shared_ptr<Node> best_child = get_max_element<std::shared_ptr<Node>>(children.begin(), UCB1_scores.begin(), UCB1_scores.end());

        if (best_child->children.size() > 0)
        {
            return best_child->traverse();
        }
        else
        {
            return best_child;
        }
    }

    // Retrieve the best child node based on number of visits (From the alpha zero pseudo code)
    std::shared_ptr<Node> Node::best_child() const
    {
        std::vector<size_t> num_visits{};
        if (children.size() == 0) return std::shared_ptr<Node>();
        for (std::shared_ptr<Node> child : children)
        {
            num_visits.push_back(child->n);
        }
        return get_max_element<std::shared_ptr<Node>>(children.begin(), num_visits.begin(), num_visits.end());
    }

    std::shared_ptr<Node> Node::softmax_sample() const {
        if (children.size() == 0) return std::shared_ptr<Node>();
        std::vector<double> distribution{};
        double exp_sum = 0;
        for (std::shared_ptr<Node> child : children)
        {
            // std::cerr << child->n << " ";
            distribution.push_back(std::exp(child->n));
            exp_sum += distribution.back();
        }
        // std::cerr << std::endl;
        for (size_t i = 0; i < distribution.size(); i++) {
            distribution[i] /= exp_sum;
            if (i > 0) {
                distribution[i] += distribution[i-1];
            }
            // std::cerr << distribution[i] << " ";
        }
        // std::cerr << std::endl;

        std::uniform_real_distribution<double> unif(0, 1);
        double random = unif(get_generator());
        size_t idx = std::lower_bound(distribution.begin(), distribution.end(), random) - distribution.begin();
        // std::cerr << random << " : " << idx << "/" << distribution.size() << std::endl; 
        return children[idx];
    }

    // Returns the labels for the policy head
    std::vector<double> Node::action_distribution(size_t num_actions)
    {
        std::vector<double> distribution(num_actions, 0.0);
        size_t tot_visits = 0;
        for (std::shared_ptr<Node> child : children)
        {
            distribution[child->action] = child->n;
            tot_visits += distribution[child->action];
        }
        for (std::shared_ptr<Node> child : children)
        {
            distribution[child->action] /= tot_visits;
        }
        return distribution;
    }

    // Get the move that gives the best child
    // Useful for baseline mcts algorithm
    chess::move Node::best_move() const
    {
        return best_child()->move;
    }

    // Get state
    chess::position Node::get_state() const
    {
        return state;
    }

    // Check if current state is a terminal state
    bool Node::is_over() const
    {
        return state.is_terminal();
    }

    // Get amount of vists
    int Node::get_n() const
    {
        return n;
    }
    chess::move Node::get_move() const
    {
        return move;
    } 
    double Node::get_value() const
    {
        return n != 0 ? t / n : 0;
    }
    void Node::make_start_node() 
    {
        is_start_node = true;
    }

    double Node::WIN_SCORE = 1.0;
    double Node::DRAW_SCORE = 0.0;
    double Node::pb_c_base = 19652;
    double Node::pb_c_init = 1.25;

}
