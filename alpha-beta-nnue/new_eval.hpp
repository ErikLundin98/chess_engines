#ifndef NEW_EVAL_H
#define NEW_EVAL_H

#include <chess/chess.hpp>

#include <limits>

namespace new_eval {


/* Constants */

const double MATERIAL_MAX = 3900;

const int END_GAME_LIMIT = 1000;

const double INF_DOUBLE = std::numeric_limits<double>::infinity();

const double MATERIAL_VALUE_MAP[5] = {
    100, // pawn
    500, // rook
    300, // knight
    300, // bishop
    900  // queen
};
const double POSITION_VALUE_MAP[7][64] = {
    {   // Pawns
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    },
    {   // Rooks
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    },
    {   // Knights
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    },
    {   // Bishops
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    },
    {   // Queens
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    },
    {   // King mid game
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20
    },
    {   // King end game
        -50,-40,-30,-20,-20,-30,-40,-50,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50
    }
};


/* Functions */

inline double evaluate(const chess::position& pos, chess::side own_side) {
    if (pos.is_checkmate()) {
        return pos.get_turn() == own_side ? -INF_DOUBLE : INF_DOUBLE; 
    } else if (pos.is_stalemate()) {
        return 0.0;
    }

    const chess::board& b = pos.pieces();

    double position_value_own = 0.0;
    double position_value_opponent = 0.0;

    double material_value_own = 0.0;
    double material_value_opponent = 0.0;

    chess::square king_sq_own = chess::square_none;
    chess::square king_sq_opponent = chess::square_none;

    /* Add values for all pieces that aren't kings */
    for (int sq_int = chess::square_a1; sq_int <= chess::square_h8; sq_int++) {
        chess::square sq = static_cast<chess::square>(sq_int);
        std::pair<chess::side, chess::piece> side_piece = b.get(sq);
        chess::side side = side_piece.first;
        chess::piece piece = side_piece.second;

        if (piece == chess::piece_king) {
            if (side == own_side) {
                king_sq_own = sq;
            } else {
                king_sq_opponent = sq;
            }

        } else if (piece != chess::piece_none) {

            /* Position value */

            // Indices for accesing value maps
            int x = sq_int % chess::files;
            int y = sq_int / chess::ranks;
            // Symmetric in x so only flip y
            if (side == chess::side_white) {
                // Maps start from h1
                y = 7 - y; // chess::ranks - 1 = 7
            }

            double position_value = POSITION_VALUE_MAP[piece][y*chess::ranks + x];

            /* Material value */
            double material_value = MATERIAL_VALUE_MAP[piece];

            if (side == own_side) {
                position_value_own += position_value;
                material_value_own += material_value;
            } else {
                position_value_opponent += position_value;
                material_value_opponent += material_value;
            }
        }
    }

    /* Add values for kings */
    // Own king
    int king_pos_map_piece_index_own = chess::piece_king + (material_value_own < END_GAME_LIMIT);
    int x_king_own = king_sq_own % chess::files;
    int y_king_own = king_sq_own / chess::ranks;
    if (own_side == chess::side_white) {
        // Maps start from h1
        y_king_own = 7 - y_king_own; // chess::ranks - 1 = 7
    }
    position_value_own += POSITION_VALUE_MAP[king_pos_map_piece_index_own][y_king_own*chess::ranks + x_king_own];
    
    // Opponent king 
    int king_pos_map_piece_index_opponent = chess::piece_king + (material_value_opponent < END_GAME_LIMIT);
    int x_king_opponent = king_sq_opponent % chess::files;
    int y_king_opponent = king_sq_opponent / chess::ranks;
    if (own_side == chess::side_black) {
        // Maps start from h1
        y_king_opponent = 7 - y_king_opponent; // chess::ranks - 1 = 7
    }
    position_value_opponent += POSITION_VALUE_MAP[king_pos_map_piece_index_opponent][y_king_opponent*chess::ranks + x_king_opponent];
    
    /* Normalize and combine material and position values */
    position_value_own = (material_value_own / MATERIAL_MAX) * position_value_own;
    double value_own = position_value_own + material_value_own;

    position_value_opponent = (material_value_opponent / MATERIAL_MAX) * position_value_opponent;
    double value_opponent = position_value_opponent + material_value_opponent;
    
    return value_own - value_opponent;
}

}
#endif // NEW_EVAL_H
