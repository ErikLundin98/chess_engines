
#include "node.hpp"
#include <chess/chess.hpp>
#include "misc.hpp"
#include "network.hpp"
#include <random>
#include <vector>
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

// Backpropagate score and visits to parent node
void Node::backpropagate(double value)
{
    t += value;
    n++;
    if (is_start_node){
        return;
    }
    if (auto p = parent.lock())
    {
        p->backpropagate(-value);
    }
}

double Node::get_terminal_value() const{
    return state.is_checkmate() ? -WIN_SCORE : DRAW_SCORE;
}

// Expand node
void Node::expand(const std::unordered_map<size_t, double>& action_probabilities)
{   
    for (auto action_prob: action_probabilities)
    {
        chess::move child_move = Network::move_from_action(action_prob.first);
        chess::position child_state = state.copy_move(child_move); // TODO - Make this optional
        std::shared_ptr<Node> new_child = std::make_shared<Node>(child_state, false, weak_from_this(), child_move);
        new_child->prior = action_prob.second;
        new_child->action = action_prob.first;
        if (new_child->is_over())
        {
            new_child->t = new_child->get_terminal_value();
            new_child->is_terminal_node = true;
            new_child->n = 1;
        }
        children.push_back(new_child);
    }
}

void Node::initialize_value(double value){
    t = value;
    n = 1;
}

void Node::add_exploration_noise(double dirichlet_alpha, double exploration_factor){
    std::default_random_engine generator;
    std::gamma_distribution<double> gamma_distribution(dirichlet_alpha, 1.0);
    for (std::shared_ptr<Node> child : children){
        double noise = gamma_distribution(generator);
        child->prior = child->prior * (1 - exploration_factor) + exploration_factor * noise;
    }
}

// Determine next node to expand/rollout by traversing tree
std::shared_ptr<Node> Node::traverse()
{
    // TODO:
    std::vector<double> UCB1_scores{};
    for (std::shared_ptr<Node> child : children)
    {
        // TODO: This should stop from traversing and ending up in terminal node
        // if (!child->is_terminal_node)
        UCB1_scores.push_back(child->UCB1());
    }

    std::shared_ptr<Node> best_child = get_max_element<std::shared_ptr<Node>>(children.begin(), UCB1_scores.begin(), UCB1_scores.end());

    if (best_child->children.size() > 0)
    {
        return best_child->children.front()->traverse();
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
    for (std::shared_ptr<Node> child : children)
    {
        num_visits.push_back(child->n);
    }
    return get_max_element<std::shared_ptr<Node>>(children.begin(), num_visits.begin(), num_visits.end());
}

std::vector<double> Node::action_distribution(size_t num_actions){
    std::vector<double> distribution(num_actions, 0.0);
    size_t tot_visits = 0;
    for (std::shared_ptr<Node> child : children){
        distribution[child->action] = child->n;
        tot_visits += distribution[child->action];
    }
    for (std::shared_ptr<Node> child : children){
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
    return is_terminal_node || state.is_checkmate() || state.is_stalemate();
}

// Get amount of vists
int Node::get_n() const
{
    return n;
}

double Node::get_value() const {
    return n != 0 ? t / n : 0.0;
}

// Print the main node and its children
std::string Node::to_string(int layers_left) const
{
    std::string tree{};
    tree += state.pieces().to_string();

    if(layers_left > 0) {
        tree += '\n' + "---children depth " + std::to_string(layers_left) + " ---\n";
        for(std::shared_ptr<Node> child_ptr : children)
        {
            tree += child_ptr->to_string(layers_left-1) + '\n';
        }

    }
    return tree;
}



double Node::WIN_SCORE = 1.0;
double Node::DRAW_SCORE = 0.0;
double Node::pb_c_base = 19652;
double Node::pb_c_init = 1.25;
    
}

