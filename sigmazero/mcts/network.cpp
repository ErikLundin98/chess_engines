#include "network.hpp"
#include <chess/chess.hpp>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <utility>
#include <set>
#define HANDCRAFTED_NETWORK
#ifdef HANDCRAFTED_NETWORK
#include <eval/eval.hpp>
#endif

namespace mcts {

Network::Evaluation Network::evaluate(chess::position& state) const{
    // Do some torch stuff
    Network::Evaluation result;
    double val = 0.5;
    std::vector<double> action_logits(1<<12, 42);
    result.value = val;
    #ifdef HANDCRAFTED_NETWORK
    {

        result.value = eval::evaluate(state);
        action_logits = std::vector<double>(action_logits.size(), 0.0);
        std::vector<chess::move> moves{state.moves()};
        for (const chess::move& move: moves) {
            chess::undo undo = state.make_move(move);
            double value = -eval::evaluate(state);
            state.undo_move(move, undo);
            size_t action = action_from_move(state, move);
            action_logits[action] = value;
        }
    }
    #endif
    // Softmax legal moves
    std::vector<chess::move> legal_moves{state.moves()};
    double exp_sum = 0.0;
    for (chess::move move: legal_moves) {
        size_t a = action_from_move(state, move);
        result.action_probabilities[a] = std::exp(action_logits[a]);
        exp_sum += result.action_probabilities[a];
    }
    // Normalize
    for (auto& kv: result.action_probabilities) {
        kv.second /= exp_sum; 
    }
    return result;
    
}


// X-Y coords -> direction index
std::pair<int, int> knight_directions[] = {
    std::make_pair(1,2),
    std::make_pair(2,1),
    std::make_pair(2,-1),
    std::make_pair(1,-2),
    std::make_pair(-1,-2),
    std::make_pair(-2,-1),
    std::make_pair(-2,1),
    std::make_pair(-1,2)
};
// X-Y coords
std::pair<int, int> queen_directions[] = {
    std::make_pair(0,1),
    std::make_pair(1,1),
    std::make_pair(1,0),
    std::make_pair(1,-1),
    std::make_pair(0,-1),
    std::make_pair(-1,-1),
    std::make_pair(-1,0),
    std::make_pair(-1,1)
};


// Regular < 0 | 3 dir bits | 3 magnitude bits / knight | 6 from square bits > 
// Promotion: < 1 | 0000 | 2 dir bits (NW, N, NE) | 2 promotion piece bits | 1 top/lower side of board bot | 3 x-value bits>
// Max value: 0b1000011111111 = 4351
// vs 64*73 = 4672 (?????)
size_t Network::action_from_move(const chess::position& state, const chess::move& move) {
    auto [side, piece] = state.pieces().get(move.from);
    int delta_x = (move.to % 8) - (move.from % 8);
    int delta_y = (move.to / 8) - (move.from / 8);
    
    // Under Promotion
    if (move.promote != chess::piece::piece_none && move.promote != chess::piece::piece_queen) {
        size_t piece = 0;
        switch(move.promote){
            case chess::piece::piece_knight: piece = 0; break;
            case chess::piece::piece_bishop: piece = 1; break;
            case chess::piece::piece_rook: piece = 2; break;
            default: std::cerr << "Underpromotion is not valid" << std::endl;
        }
        size_t dir = delta_x + 1; // (-1, 0, 1) -> (0, 1, 2)
        size_t underprom_idx = dir*3 + piece;
        return 1 << (6+3+3) | underprom_idx << (6);
    }
    size_t action = (size_t) move.from;
    auto get_dir_idx = [](std::pair<int, int> arr[], std::pair<int, int> item){
        auto it = std::find(arr, arr+8, item);
        if (it == arr+8) {
            std::cout << "Too big index" << std::endl;
        }
        return std::distance(arr, it);
    };
    // KNIGHT
    if (piece == chess::piece::piece_knight){
        std::pair<int, int> dir = std::make_pair(delta_x, delta_y);
        size_t dir_idx = get_dir_idx(knight_directions, dir);
        return dir_idx<<(6+3) | 0<<(6) | action;
    }
    // ALL ELSE
    int dir_x = delta_x != 0 ? delta_x / std::abs(delta_x) : 0;
    int dir_y = delta_y != 0 ? delta_y / std::abs(delta_y) : 0;
    std::pair<int, int> dir = std::make_pair(dir_x, dir_y);
    size_t dir_idx = get_dir_idx(queen_directions, dir);
    size_t magnitude = std::abs(delta_x);

    return magnitude << (6+3) | dir_idx << (6) | action;

}



chess::move Network::move_from_action(const chess::position& state, size_t action) {
    chess::square from =  static_cast<chess::square>(action & ((1 << 6) - 1) );
    int x = from % 8;
    int y = from / 8;
    bool underpromotion = 1 << (6+3+3) & action;
    if (underpromotion) {
        int dy = y == 1 ? -1 : 1;
        size_t underprom_idx = ((1<<4) - 1) << 6;
        underprom_idx >>= 6;
        int dir = underprom_idx / 3;
        int dx = dir - 1;
        int piece_idx = underprom_idx % 3;
        chess::piece piece;
        switch(piece_idx){
            case 0: piece = chess::piece::piece_knight; break;
            case 1: piece = chess::piece::piece_bishop; break;
            default: piece = chess::piece::piece_rook; 
        }
        chess::square to = static_cast<chess::square>(x+dx + (y+dy)*8);
        chess::move move{from, to};
        move.promote = piece;
        return move;
    }
    size_t magnitude = ((1<<3) - 1) << (6) & action;
    magnitude >>= 6;
    size_t dir = ((1<<3) - 1) << (6+3) & action;
    dir >>= 6+3;
    // KNIGHT
    if (magnitude == 0) {
        auto [dx, dy] = knight_directions[dir];
        chess::square to = static_cast<chess::square>(x + dx + (y + dy)*8);
        return chess::move{from, to};
    }

    std::pair<int, int> xy_dir = queen_directions[dir];
    int to_x = x + xy_dir.first * magnitude;
    int to_y = y + xy_dir.second * magnitude;
    chess::square to = static_cast<chess::square>(to_x + to_y*8);
    chess::move move{from, to};

    auto [side, piece] = state.pieces().get(from); 
    if (piece == chess::piece::piece_pawn && (to_y == 0 || to_y == 7)) {
        move.promote = chess::piece::piece_queen;
    }
    return move;
}

}