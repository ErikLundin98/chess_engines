#include <fstream>
#include <sstream>
#include <string>
#include <forward_list>
#include <vector>
#include <tuple>
#include <utility>
#include <algorithm>
#include <iostream>
#include <filesystem>

#include <torch/torch.h>
#include <chess/chess.hpp>
#include <sigmazero/drl/sigmanet.hpp>
#include <sigmazero/drl/action_encodings.hpp>

/*
Load dataset from .csv file
parse into list<pair<int, vector<chess::move>>>
convert chess::moves to policy logits by using sigmanet functions
save as a dataset
*/

int main()
{
    chess::init();
    sigmanet model(0, 64, 10);
    //std::forward_list<feature_label_pair> data;

    std::vector<torch::Tensor> features;
    std::vector<torch::Tensor> policies;
    std::vector<torch::Tensor> values;

    // Read file
    std::string fname = "../sigmazero/verifications/uci_dataset.csv";
    std::ifstream f(fname);
    std::string line;
    int file_size = 3522926;
    int current = 0;
    int start_game = 0;
    int n_saves = 0;
    std::string feature_n = "features.pt";
    std::string policy_n = "policies.pt";
    std::string value_n = "values.pt";
    while(std::getline(f, line))
    {   
        if(current++ < start_game) continue;
        if(current % 2500 == 0 && current != start_game) {
            n_saves++;
            if(n_saves == 4) 
            {
                n_saves = 0;
                feature_n = "I" + feature_n;
                policy_n = "I" + policy_n;
                value_n = "I" + value_n;
            }
            std::cout << current << "/" << file_size << "  " << std::endl;
            std::cout << "appending & saving data..." << std::endl;

            torch::Tensor feature_stack;
            torch::Tensor policy_stack;
            torch::Tensor value_stack;
            bool loaded = false;
            try
            {
                torch::load(feature_stack, std::filesystem::path(feature_n));
                torch::load(policy_stack, std::filesystem::path(policy_n));
                torch::load(value_stack, std::filesystem::path(value_n));
                loaded = true;
                std::cout << "loaded tensors with size:" << std::endl << feature_stack.sizes() << std::endl << policy_stack.sizes() << std::endl << value_stack.sizes() << std::endl;
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << std::endl;
            }
            if(loaded)
            {
                feature_stack = torch::cat({feature_stack, torch::stack(features)});
                policy_stack = torch::cat({policy_stack, torch::stack(policies)});
                value_stack = torch::cat({value_stack, torch::stack(values)});
            }
            else {
                feature_stack = torch::stack(features);
                policy_stack = torch::stack(policies);
                value_stack = torch::stack(values);
            }

            features.clear();
            policies.clear();
            values.clear();

            torch::save(feature_stack, std::filesystem::path(feature_n));
            torch::save(policy_stack, std::filesystem::path(policy_n));
            torch::save(value_stack, std::filesystem::path(value_n));
        }
       
        // Line contains line from csv file
        std::remove_if(line.begin(), line.end(), isspace);
        size_t delim_pos = line.find(';');
        std::string winner = line.substr(0, delim_pos);
        std::string lan_moves = line.substr(delim_pos+1, line.size());
        chess::side winning_side = chess::side_none;
        if(winner == "w") winning_side = chess::side_white;
        else if(winner == "b") winning_side = chess::side_black;
        // Iterate over game to create features
        std::istringstream iss(lan_moves);
        std::string move;
        chess::position p = chess::position::from_fen(chess::position::fen_start);
        while(std::getline(iss, move, ','))
        {
            //std::cout << "read move " << move << std::endl;
            int value = 0;
            if(winning_side != chess::side_none) value = p.get_turn() == winning_side ? 1 : -1;
            chess::move policy = chess::move::from_lan(move);

            features.push_back(model->encode_input(p));
            size_t policy_index = action_encodings::action_from_move(policy);
            policy_index = action_encodings::cond_flip_action(p, policy_index); // one hot decode after instead
            // std::vector<double> policy_vector(64*73);
            // policy_vector[policy_index] = 1.0;

            policies.push_back(torch::tensor(static_cast<int>(policy_index)));
            values.push_back(torch::tensor(value));

            p.make_move(policy);
        }
    }
    
}
