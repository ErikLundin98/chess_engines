#ifndef EVAL_MATERIAL_H
#define EVAL_MATERIAL_H
#include <chess/chess.hpp>

namespace eval {
    const int MATERIAL_VALUES [5] = {
        100, // pawn
        500, // rook
        300, // knight
        300, // bishop
        900  // queen
    };
}

#endif // EVAL_MATERIAL_H