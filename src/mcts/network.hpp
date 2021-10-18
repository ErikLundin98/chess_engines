#ifndef NETWORK_H
#define NETWORK_H

#include <chess/chess.hpp>
#include <unordered_map>
#include <algorithm>

struct Network {
    struct Evaluation {
        double value;
        std::unordered_map<size_t, double> action_probabilities;
    };
    
    Evaluation evaluate(chess::position& state){
        // Do some torch stuff
        Evaluation result;
        double val;
        vector<double> action_logits;
        
        
        result.value = val;
        // Softmax legal moves
        std::vector<chess::move> legal_moves{state.moves()};
        double exp_sum = 0.0;
        for (chess::move move: legal_moves) {
            size_t a = action_from_move(move);
            result.action_probabilities[a] = exp(action_logits[a]);
            exp_sum += result.action_probabilities[a];
        }
        // Normalize
        for (auto& kv: result.action_probabilities.) {
            kv.second /= exp_sum; 
        }
        return result;
        
    }

    // SHOULD EXIST SIMPLER WAY, DO LATER
    static size_t action_from_move(chess::position state, chess::move& move) {
        return move.from + (move.to << 6);
        // // <pos_on_board..> | <x_delta..> | <y_delta..> | <knight_moves> | <underprom_type..>
        // const knight[][] knight_yx = {}
        // size_t res = 0;
        // size_t res_idx = 0;
        // chess::piece piece; // how to check if is pawn
        // auto [side, piece] = state.pieces().get(move.from);
        // if (piece == chess::piece::piece_pawn) {

        // }
        // res |= (size_t) move.from;
        // res_idx += 6;



        // int delta_x = (move.to % 8) - (move.from % 8);
        // int delta_y = (move.to / 8) - (move.from / 8);
        // if (delta_x == delta_y || delta_x == 0 || delta_y == 0) {
        //     res |= (delta_x + 7) << res_idx; // 0...14
        //     res_idx += 4;
        //     res |= (delta_y + 7) << res_idx;
        //     res_idx += 4;
        //     // Queen Move
        // } else {
        //     // Knight move
        // }
        // // 
        // int delta_x = (delta+7) % 14;
        // int delta_y = (delta+7) % 14;
        // // {-7..7, -7..7} ex {0, 0}
        // return location_idx | (delta << 6);

    }



    static chess::move move_from_action(size_t action) {
        size_t to_val = action>>6;
        chess::square from(action ^ (to_val << 6));
        chess::square to(to_val);
        chess::move move{from, to};
        return move;
    }



};





#endif // NETWORK_H