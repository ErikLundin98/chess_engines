#include "NNUE.hpp"

NNUE::evaluator::evaluator(const std::string path) {
    torch::load(l0_weights, path + "input_layer.weight.pt");
    torch::load(l0_biases, path + "input_layer.bias.pt");
    torch::load(l1_weights, path + "fc1.weight.pt");
    torch::load(l1_biases, path + "fc1.bias.pt");
    torch::load(l2_weights, path + "fc2.weight.pt");
    torch::load(l2_biases, path + "fc2.bias.pt");
    torch::load(l3_weights, path + "fc3.weight.pt");
    torch::load(l3_biases, path + "fc3.bias.pt");

}

float NNUE::evaluator::forward(torch::Tensor x) {
    auto f1 = torch::matmul(l1_weights,x).squeeze() + l1_biases;

    auto z1 = torch::relu(f1);

    auto f2 = torch::matmul(l2_weights,z1).squeeze() + l2_biases;
    auto z2 = torch::relu(f2);

    auto output = torch::matmul(l3_weights,z2).squeeze() + l3_biases;

    return output[0].item<float>();
}

void NNUE::accumulator::refresh(const evaluator& eval, enum perspective perspective, const chess::position & pos) {
    
    torch::Tensor encoding = torch::zeros(64*64*10);
    halfkp_encode(encoding, pos, perspective);
    encoding = encoding.unsqueeze(1);

    if(perspective == white) {
        accumulator_white = torch::matmul(eval.l0_weights,encoding).squeeze() + eval.l0_biases;
    }
    else {
        accumulator_black = torch::matmul(eval.l0_weights,encoding).squeeze() + eval.l0_biases;
    }
}

void NNUE::accumulator::update(const evaluator& eval, enum perspective perspective, const chess::move& move) {
    return;
}

void NNUE::accumulator::print_accumulator(enum perspective perspective) {
    if(perspective == white) {
        std::cout << accumulator_white << std::endl;
    }
    else {
        std::cout << accumulator_black << std::endl;
    }
    
}

chess::bitboard NNUE::accumulator::bitboard_mirror(chess::bitboard bb) {
    chess::bitboard flipped_bb = 0;
    for (int i = 0; i < chess::files; i++) {
        flipped_bb |= ((((bb >> (((chess::ranks - 1) - i) * chess::files))) & 255) << i * chess::files);
    }

    return flipped_bb;
}

void NNUE::accumulator::halfkp_encode(torch::Tensor& features, const chess::position & pos, enum perspective perspective) {
    //chess::position pos = chess::position::from_fen(fen_string); 
    const chess::board& board = pos.pieces();

    // bitboards for kings
    chess::bitboard bb_wk = board.piece_set(chess::piece_king, chess::side_white);
    chess::bitboard bb_bk = board.piece_set(chess::piece_king, chess::side_black);

    // bitboards for all non king pieces
    std::vector<chess::bitboard> bitboards;
    bitboards.push_back(board.piece_set(chess::piece_pawn, chess::side_white));
    bitboards.push_back(board.piece_set(chess::piece_knight, chess::side_white));
    bitboards.push_back(board.piece_set(chess::piece_bishop, chess::side_white));
    bitboards.push_back(board.piece_set(chess::piece_rook, chess::side_white));
    bitboards.push_back(board.piece_set(chess::piece_queen, chess::side_white));
    bitboards.push_back(board.piece_set(chess::piece_pawn, chess::side_black));
    bitboards.push_back(board.piece_set(chess::piece_knight, chess::side_black));
    bitboards.push_back(board.piece_set(chess::piece_bishop, chess::side_black));
    bitboards.push_back(board.piece_set(chess::piece_rook, chess::side_black));
    bitboards.push_back(board.piece_set(chess::piece_queen, chess::side_black));

    int halfkp_idx = 0;

    if(pos.get_turn() == chess::side_white) {
        if(perspective == white) {
            // white king
            for(int king_sq = 0; king_sq < 64; king_sq++) {
                if(((bb_wk >> king_sq) & 1) == 0) {
                    halfkp_idx += 64 * 10;
                    continue;
                }
                
                // white pieces
                for(int p = 0; p < 5; p++) {
                    for(int piece_sq = 0; piece_sq < 64; piece_sq++) {
                        if(((bitboards[p] >> piece_sq) & 1) == 1) {
                            features[halfkp_idx] = 1;
                        }

                        halfkp_idx++;
                    }
                }

                // black pieces
                for(int p = 5; p < 10; p++) {
                    for(int piece_sq = 0; piece_sq < 64; piece_sq++) {
                        if(((bitboards[p] >> piece_sq) & 1) == 1) {
                            features[halfkp_idx] = 1;
                        }

                        halfkp_idx++;
                    }
                }
            }
            return;
        }
        // black perspective
        else {
            // black king
            for(int king_sq = 0; king_sq < 64; king_sq++) {
                if(((bitboard_mirror(bb_bk) >> king_sq) & 1) == 0) {
                    halfkp_idx += 64 * 10;
                    continue;
                }
                            
                // black pieces
                for(int p = 5; p < 10; p++) {
                    for(int piece_sq = 0; piece_sq < 64; piece_sq++) {
                        if(((bitboard_mirror(bitboards[p]) >> piece_sq) & 1) == 1) {
                            features[halfkp_idx] = 1;
                        }

                        halfkp_idx++;
                    }
                }

                // white pieces
                for(int p = 0; p < 5; p++) {
                    for(int piece_sq = 0; piece_sq < 64; piece_sq++) {
                        if(((bitboard_mirror(bitboards[p]) >> piece_sq) & 1) == 1) {
                            features[halfkp_idx] = 1;
                        }

                        halfkp_idx++;
                    }
                }
            }
            return;
        }
    }   
    // Black turn
    else {
        if(perspective == black) {
            // black king
            for(int king_sq = 0; king_sq < 64; king_sq++) {
                if(((bitboard_mirror(bb_bk) >> king_sq) & 1) == 0) {
                    halfkp_idx += 64 * 10;
                    continue;
                }
                
                // black pieces
                for(int p = 5; p < 10; p++) {
                    for(int piece_sq = 0; piece_sq < 64; piece_sq++) {
                        if(((bitboard_mirror(bitboards[p]) >> piece_sq) & 1) == 1) {
                            features[halfkp_idx] = 1;
                        }

                        halfkp_idx++;
                    }
                }

                // white pieces
                for(int p = 0; p < 5; p++) {
                    for(int piece_sq = 0; piece_sq < 64; piece_sq++) {
                        if(((bitboard_mirror(bitboards[p]) >> piece_sq) & 1) == 1) {
                            features[halfkp_idx] = 1;
                        }

                        halfkp_idx++;
                    }
                }
            }
            return;
        }
        // white persepctive
        else {
            // white king
            for(int king_sq = 0; king_sq < 64; king_sq++) {
                if(((bb_wk >> king_sq) & 1) == 0) {
                    halfkp_idx += 64 * 10;
                    continue;
                }
                
                // white pieces
                for(int p = 0; p < 5; p++) {
                    for(int piece_sq = 0; piece_sq < 64; piece_sq++) {
                        if(((bitboards[p] >> piece_sq) & 1) == 1) {
                            features[halfkp_idx] = 1;
                        }

                        halfkp_idx++;
                    }
                }

                // black pieces
                for(int p = 5; p < 10; p++) {
                    for(int piece_sq = 0; piece_sq < 64; piece_sq++) {
                        if(((bitboards[p] >> piece_sq) & 1) == 1) {
                            features[halfkp_idx] = 1;
                        }

                        halfkp_idx++;
                    }
                }
            }
            return;
        }
    }
}


