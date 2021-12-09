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

    float forward(torch::Tensor accumulator_white, torch::Tensor accumulator_black, chess::side turn);
    torch::Tensor l0_weights, l1_weights, l2_weights, l3_weights, l0_biases, l1_biases, l2_biases, l3_biases;

private:

    
};

class accumulator {
public:
    accumulator() = default;
    accumulator(const accumulator& other);

    void update(const evaluator& eval, enum perspective perspective, const chess::move& move,  const chess::position& position);
    void refresh(const evaluator& eval, enum perspective perspective, const chess::position& pos);
    void print_accumulator(enum perspective perspective);
    
    torch::Tensor accumulator_white = torch::zeros(M);
    torch::Tensor accumulator_black = torch::zeros(M);

private:
    chess::square white_king_pos{chess::square_e1};
    chess::square black_king_pos{chess::square_e8};
    
    chess::bitboard bitboard_mirror(chess::bitboard bb);

    void halfkp_encode(torch::Tensor& result, const chess::position & pos, enum perspective perspective);

    inline int get_halfkp_idx(const chess::piece& piece_type, const chess::square& piece_square, const chess::square& king_square, const chess::side& side) { 
        return 640*king_square + 320*side + 64*map_piece_idx(piece_type) + piece_square; 
    }

    inline int reverse_idx(int idx) {return 8*(7 - (idx / 8)) + idx % 8;}

    inline int map_piece_idx(const chess::piece& piece) {
        switch(piece) {
            case chess::piece_pawn:
                return 0;
            case chess::piece_rook:
                return 3;
            case chess::piece_knight:
                return 1;
            case chess::piece_bishop:
                return 2;
            case chess::piece_queen:
                return 4;
            case chess::piece_king:
                return 5;
            default: 
                return -1;
        }
    }
};
}


#endif //NNUE_H


