#ifndef NNUE_H
#define NNUE_H

#include <torch/torch.h>
#include <string>
#include <chess/chess.hpp>

#define M 256

namespace NNUE
{

enum perspective {
    white,
    black
};

class evaluator {
public:
    evaluator(const std::string path);

    float forward(torch::Tensor x);
    torch::Tensor l0_weights, l1_weights, l2_weights, l3_weights, l0_biases, l1_biases, l2_biases, l3_biases;

private:

    
};

class accumulator {
public:
    accumulator() = default;
    void update(const evaluator& eval, enum perspective perspective, const chess::move& move);
    void refresh(const evaluator& eval, enum perspective perspective, const chess::position& pos);
    void print_accumulator(enum perspective perspective);
    
    torch::Tensor accumulator_white = torch::zeros(M);
    torch::Tensor accumulator_black = torch::zeros(M);

private:
    

    chess::bitboard bitboard_mirror(chess::bitboard bb);
    void halfkp_encode(torch::Tensor& result, const chess::position & pos, enum perspective perspective);
    

};

}


#endif //NNUE_H


