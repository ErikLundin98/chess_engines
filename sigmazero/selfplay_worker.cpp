#include "selfplay_worker.hpp"

#include <sstream>

#include "base64.hpp"

selfplay_worker::selfplay_worker()
{
    main_node = std::make_shared<mcts::Node>(game.get_position());
}

selfplay_worker::selfplay_worker(chess::game game) : game{game}
{
    main_node = std::make_shared<mcts::Node>(game.get_position());
}

const chess::position& selfplay_worker::get_position() const
{
    return game.get_position();
}

const chess::game& selfplay_worker::get_game() const
{
    return game;
}

bool selfplay_worker::game_is_terminal(size_t max_game_size) const
{
    return game.get_position().is_terminal() || game.size() >= max_game_size;
}

std::size_t selfplay_worker::replay_size() const
{
    return images.size();
}

void selfplay_worker::initial_setup(const std::pair<double, std::unordered_map<size_t, double>>& evaluation)
{
    main_node->explore_and_set_priors(evaluation);
    main_node->add_exploration_noise(0.3, 0.25);
}

chess::position selfplay_worker::traverse()
{
    current_node = main_node->traverse();
    
    // if(current_node->is_over())
    // {
    //     current_node->backpropagate(current_node->get_terminal_value());
    //     return std::nullopt;
    // }

    return current_node->get_state();
}

void selfplay_worker::explore_and_set_priors(const std::pair<double, std::unordered_map<size_t, double>>& evaluation)
{
    if(!current_node->is_over())
    {
        current_node->expand(evaluation.second);
    }
    current_node->backpropagate(evaluation.first);
}

chess::move selfplay_worker::make_best_move(torch::Tensor position_encoding, bool record)
{
    if(record)
    {
        images.push_back(position_encoding);
        std::vector<double> action_dist = main_node->action_distribution();
        torch::Tensor action_tensor = torch::tensor(action_dist);
        policies.push_back(action_tensor);
        players.push_back(game.get_position().get_turn());
    }

    main_node = main_node->best_child();
    main_node->make_start_node();
    chess::move best_move = main_node->get_move();
    game.push(best_move);
    return best_move;
}

void selfplay_worker::output_game(std::ostream& stream)
{
    // send tensors
    for(size_t i = 0 ; i < images.size(); ++i)
    {
        std::optional<int> game_value = game.get_value(players[i]);
        float terminal_value = game_value ? *game_value : 0.0f;
        torch::Tensor value = torch::tensor(terminal_value);

        try
        {
            stream << encode(images[i]) << ' ' << encode(value) << ' ' << encode(policies[i]) << std::endl; //according to side
        }
        catch(const std::exception& e)
        {
            std::cerr << "exception raised when encoding and sending game tensors, skipping..." << std::endl;
        }
    }
}

void selfplay_worker::reinit_main_node() 
{
    main_node = std::make_shared<mcts::Node>(game.get_position());
}

std::string selfplay_worker::encode(const torch::Tensor &tensor)
{
	std::ostringstream data;
	torch::save(tensor, data);
	return base64_encode(data.str());
}