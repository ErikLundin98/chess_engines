#include "NNUE.hpp"

using namespace torch::indexing;

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

float NNUE::evaluator::forward(torch::Tensor accumulator_white, torch::Tensor accumulator_black, chess::side turn) {

    torch::Tensor x;

    if(turn == chess::side_black) {
		x = torch::cat({accumulator_black.unsqueeze(1), accumulator_white.unsqueeze(1)}, 0);
	}
	else {
		x = torch::cat({accumulator_white.unsqueeze(1), accumulator_black.unsqueeze(1)}, 0);
	}

    auto z0 = torch::relu(x);
    auto f1 = torch::matmul(l1_weights, z0).squeeze() + l1_biases;

    auto z1 = torch::relu(f1);

    auto f2 = torch::matmul(l2_weights, z1).squeeze() + l2_biases;
    auto z2 = torch::relu(f2);

    auto output = torch::matmul(l3_weights, z2).squeeze() + l3_biases;

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

    const chess::board& board = pos.pieces();

    chess::bitboard bb_wk = board.piece_set(chess::piece_king, chess::side_white);
    chess::bitboard bb_bk = board.piece_set(chess::piece_king, chess::side_black);

    for(int i = 0; i < 64; i++) {
        if( ((bb_wk >> i) & 1) == 1) {
            white_king_pos = (chess::square)i;
        }
        if( ((bb_bk >> i) & 1) == 1) {
            black_king_pos = (chess::square)i;
        }
    }
}

int NNUE::accumulator::get_halfkp_idx(const chess::piece& piece_type, const chess::square& piece_square, const chess::square& king_square, const chess::side& side) {
    int p_idx = piece_type * chess::sides + side;
    int halfkp_idx = (king_square * 10 + p_idx) * chess::squares + piece_square;
}

void NNUE::accumulator::update(const evaluator& eval, enum perspective perspective, const chess::move& move, const chess::board& board) {
    std::pair<chess::side, chess::piece> moved_piece = board.get(move.from);
    std::pair<chess::side, chess::piece> captured_piece = board.get(move.to);

    int idx;

    if(perspective == white) {
        // Remove moved piece
        idx = get_halfkp_idx(moved_piece.second, move.from, white_king_pos, moved_piece.first);
        accumulator_white -= eval.l0_weights.index({Slice(), idx});

        // Add new position (check if promoted)
        chess::piece new_piece = moved_piece.second;
        if (move.promote != chess::piece_none) {
            new_piece = move.promote;
        }

        idx = get_halfkp_idx(new_piece, move.to, white_king_pos, moved_piece.first);
        accumulator_white += eval.l0_weights.index({Slice(), idx});

        // Remove taken
        if (captured_piece.second != chess::piece_none) {
            idx = get_halfkp_idx(captured_piece.second, move.to, white_king_pos, captured_piece.first);
            accumulator_white -= eval.l0_weights.index({Slice(), idx});
        }
    } 
   else {
        // Remove moved piece
        idx = get_halfkp_idx(moved_piece.second, (chess::square)reverse_idx(move.from), (chess::square)reverse_idx(black_king_pos), moved_piece.first);
        accumulator_white -= eval.l0_weights.index({Slice(), idx});

        // Add new position (check if promoted)
        chess::piece new_piece = moved_piece.second;
        if (move.promote != chess::piece_none) {
            new_piece = move.promote;
        }

        idx = get_halfkp_idx(new_piece, (chess::square)reverse_idx(move.to), (chess::square)reverse_idx(white_king_pos), moved_piece.first);
        accumulator_white += eval.l0_weights.index({Slice(), idx});

        // Remove taken
        if (captured_piece.second != chess::piece_none) {
            idx = get_halfkp_idx(captured_piece.second, (chess::square)reverse_idx(move.to), (chess::square)reverse_idx(white_king_pos), captured_piece.first);
            accumulator_white -= eval.l0_weights.index({Slice(), idx});
        }
    }     
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


