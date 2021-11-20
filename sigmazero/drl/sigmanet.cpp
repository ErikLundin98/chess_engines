#include <algorithm>
#include <cstdint>
#include <cmath>
#include <chess/chess.hpp>
#include <iostream>
#include "action_encodings.hpp"
#include "sigmanet.hpp"

residual_block::residual_block(int filters) {

    conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, filters, 3).padding(1));
    batchnorm1 = torch::nn::BatchNorm2d(filters);

    conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, filters, 3).padding(1));
    batchnorm2 = torch::nn::BatchNorm2d(filters);

    register_module("conv1", conv1);
    register_module("batchnorm1", batchnorm1);
    register_module("conv2", conv2);
    register_module("batchnorm2", batchnorm2);
}

torch::Tensor residual_block::forward(torch::Tensor x) {

    torch::Tensor y = x;

    x = conv1->forward(x);
    x = batchnorm1->forward(x);

    x = torch::relu(x);

    x = conv2->forward(x);
    x = batchnorm2->forward(x);

    x = y + x;
    x = torch::relu(x);

    return x;
}
// History unused currently
sigmanet_impl::sigmanet_impl(int history, int filters, int blocks) : history{history}, in_channels{1*feature_planes + constant_planes}, filters{filters}, blocks{blocks} {

    input_conv = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, filters, 3).stride(1).padding(1)),
        torch::nn::BatchNorm2d(filters),
        torch::nn::ReLU()
    );

    residual = torch::nn::Sequential();

    for (int i = 0; i < blocks; i++) {
        residual->push_back(residual_block(filters));
    }

    value_head = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, 1, 1)),
        torch::nn::BatchNorm2d(1),
        torch::nn::ReLU(),
        torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(-2).end_dim(-1)),
        torch::nn::Linear(8 * 8, 256),
        torch::nn::ReLU(),
        torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(-2).end_dim(-1)),
        torch::nn::Linear(256, 1),
        torch::nn::Tanh()
    );

    policy_head = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, 2, 1)),
        torch::nn::BatchNorm2d(2),
        torch::nn::ReLU(),
        torch::nn::Flatten(torch::nn::FlattenOptions().start_dim(-3).end_dim(-1)),
        torch::nn::Linear(2 * 8 * 8, 8 * 8 * 73)
    );

    register_module("input_conv", input_conv);
    register_module("residual", residual);
    register_module("value_head", value_head);
    register_module("policy_head", policy_head);
}


std::pair<torch::Tensor, torch::Tensor> sigmanet_impl::forward(torch::Tensor x) {
    
    x = input_conv->forward(x);
    x = residual->forward(x);

    auto value = value_head->forward(x);
    auto policy = torch::softmax(policy_head->forward(x), -1);

    return std::make_pair(value, policy);
}

// Assumes that model is in eval mode
//TODO: Hash chess::move???????
std::pair<double, std::unordered_map<size_t, double>> sigmanet_impl::evaluate(const chess::position& p, torch::Device device)
{
    auto[value, policy_logits] = forward(encode_input(p).unsqueeze(0).to(device));
    // policy now is a 4672x1 tensor of logits
    // Value is a 1x1 tensor of a policy
    return decode_output(policy_logits, value, p);
    // !IMPORTANT: when passing values to network, pass according to player side
}

std::pair<double, std::unordered_map<size_t, double>> sigmanet_impl::decode_output(const torch::Tensor& policy, torch::Tensor value, const chess::position& p) const {
    return std::make_pair(value.item<double>(), valid_policy_probabilities(policy, p));
}

std::unordered_map<size_t, double> sigmanet_impl::valid_policy_probabilities(const torch::Tensor& policy_logits, const chess::position& state) const {
        
    // Softmax legal moves
    std::unordered_map<size_t, double> policy_probabilities;
    std::vector<chess::move> legal_moves{state.moves()};
    double exp_sum = 0.0;
    for (chess::move move: legal_moves) {
        size_t a = action_encodings::action_from_move(move);
        double value = policy_logits[0][a].item<double>();
        policy_probabilities[a] = std::exp(value);
        exp_sum += policy_probabilities[a];
    }
    // Normalize
    for (auto& kv: policy_probabilities) {
        kv.second /= exp_sum; 
    }

    return policy_probabilities;
};

torch::Tensor sigmanet_impl::encode_input(const chess::position& pos) const
{
    using namespace torch::indexing;

    const int planes = 12+7;
    torch::Tensor input = torch::zeros({planes, 8, 8});

    int j = 0;
    chess::side p1 = pos.get_turn();
    chess::side p2 = chess::opponent(p1);

    // feature planes
    bool flip = p1 == chess::side_black;
    // p1 pieces
    for(int p = chess::piece_pawn; p <= chess::piece_king; p++)
    {
        chess::bitboard bb = pos.get_board().piece_set(static_cast<chess::piece>(p), p1);
        torch::Tensor plane = bitboard_plane(bb);
        if(flip) plane = torch::flipud(plane);

        input.index_put_({j++}, plane);
    }

    // p2 pieces
    for(int p = chess::piece_pawn; p <= chess::piece_king; p++)
    {
        chess::bitboard bb = pos.get_board().piece_set(static_cast<chess::piece>(p), p2);
        torch::Tensor plane = bitboard_plane(bb);
        if(flip) plane = torch::flipud(plane);

        input.index_put_({j++}, plane);
    }

    // constant planes

    // color
    input.index_put_({j++}, static_cast<int>(p1));

    // move count
    input.index_put_({j++}, pos.get_fullmove());

    // p1 castling
    input.index_put_({j++}, pos.can_castle_kingside(p1));
    input.index_put_({j++}, pos.can_castle_queenside(p1));

    // p2 castling
    input.index_put_({j++}, pos.can_castle_kingside(p2));
    input.index_put_({j++}, pos.can_castle_queenside(p2));

    // no-progress count
    input.index_put_({j++}, pos.get_halfmove_clock());

    return input;

}

int sigmanet_impl::get_input_channels() const
{
    return in_channels;
}

torch::Tensor sigma_loss(torch::Tensor z, torch::Tensor v, torch::Tensor p, torch::Tensor pi) {

    p = torch::add(p, 1e-8);
    torch::Tensor value_loss = torch::sum(torch::mul(z-v, z-v));
    torch::Tensor policy_loss = -torch::sum(torch::mul(pi, torch::log(p)));
    torch::Tensor loss = value_loss + policy_loss;
    return loss;
}


torch::Tensor bitboard_plane(chess::bitboard bb)
{
    torch::Tensor plane = torch::zeros({8, 8});

    for(chess::square sq: chess::set_elements(bb))
    {
        chess::file f = chess::file_of(sq);
        chess::rank r = chess::rank_of(sq);

        using namespace torch::indexing;
        plane.index_put_({static_cast<int>(r), static_cast<int>(f)}, 1.0f);
    }

    return plane;
}
