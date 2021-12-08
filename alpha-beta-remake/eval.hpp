#ifndef EVAL_H
#define EVAL_H

#include <chess/chess.hpp>
#include "material.hpp"
#include "piece_maps.hpp"
#include <iostream>
#include <vector>
#include <limits>


namespace eval {

using namespace chess;

class evaluator {
private:
    const position& state;
    chess::side side;
    double position_value = 0.0;
    double material_value = 0.0;
public:
    static constexpr int MATERIAL_MAX = 3900;
    static constexpr int END_GAME_LIMIT = 1000;
    static double infinity(){
        return std::numeric_limits<double>::infinity();
    }

    evaluator(const chess::position& state, chess::side side) : state{state}, side{side} {}

    double eval_side(){
        add_piece_values(piece::piece_pawn);
        add_piece_values(piece::piece_rook);
        add_piece_values(piece::piece_knight);
        add_piece_values(piece::piece_bishop);
        add_piece_values(piece::piece_queen);
        add_king_values();
        position_value = (material_value / MATERIAL_MAX) * position_value;
        double value = position_value + material_value;
        return value ;
    }
private:

    void add_piece_values(chess::piece piece) {
        chess::bitboard piece_set = state.pieces().piece_set(piece, side);
        std::vector<square> piece_locations = set_elements(piece_set);

        for(square location: piece_locations) {
            position_value += eval::maps::get_value(eval::maps::pieces[piece], location, side);
            material_value += MATERIAL_VALUES[piece];
            
        }
    }

    void add_king_values(){
        chess::bitboard king_set = state.pieces().piece_set(chess::piece_king, side);
        std::vector<square> king_locations = set_elements(king_set);
        square king_location = king_locations[0];
        if (material_value >= END_GAME_LIMIT) {
            position_value += eval::maps::get_value(eval::maps::pieces[5], king_location, side);
        } else {
            position_value += eval::maps::get_value(eval::maps::pieces[6], king_location, side);
        }
    }
};

/// Evaluates a state and returns an estimated value of the state for the player whos turn it is.
///
/// \param state The chess board state.
/// \returns Value of state. For a win or a loss, the value is infinite and negativily infinite respectively.
inline double evaluate(const chess::position& state){
    
    if(state.is_checkmate()) {
        return state.get_turn() == chess::side_white ? -evaluator::infinity() : evaluator::infinity();
    }
    else if(state.is_stalemate()) {
        return 0.0;
    }
    
    evaluator cur(state, chess::side_white);
    evaluator other(state, chess::side_black);
    return cur.eval_side() - other.eval_side();
}


}

#endif // EVAL_H