#include "minmax.hpp"

namespace minmax {
    const double value_map[] = {1.0, 5.0, 3.0, 3.0, 9.0, 0.0};
    double estimate_score(const chess::position &p, const chess::side own_side) 
    {

        const chess::board& b = p.pieces();
        
        double score = 0;
        for (int sq_int = chess::square_a1; sq_int <= chess::square_h8; sq_int++) {
            chess::square sq = static_cast<chess::square>(sq_int);
            std::pair<chess::side, chess::piece> side_piece = b.get(sq);
            chess::side side = side_piece.first;
            chess::piece piece = side_piece.second;
        
            if (piece != chess::piece_none) {
                double piece_score = value_map[piece];

                if (side == own_side) {
                    score += piece_score;
                } else {
                    score -= piece_score;
                }
            }
        }
        return score;
    }


    bool sort_ascending(const std::pair<double, chess::position>& p1, const std::pair<double, chess::position>& p2)
    {
    return p1.first < p2.first;
    }

    bool sort_descending(const std::pair<double, chess::position>& p1, const std::pair<double, chess::position>& p2)
    {
    return p1.first > p2.first;
    }

    double rec_minmax(const chess::position& pos, bool max_node, double alpha, double beta, int depth_left, const chess::side own_side, std::unordered_map<size_t, double>& pos_scores, uci::search_info& info, const std::atomic_bool& stop, const std::chrono::steady_clock::time_point& start_time, float max_time) 
    {
        size_t pos_hash = pos.hash();

        double pos_val;

        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_time = current_time - start_time;

        if (pos_scores.count(pos_hash) >= 1) {              
            pos_val = pos_scores[pos_hash];

        } else if (pos.is_checkmate()) {
            if (pos.get_turn() == own_side) {
                pos_val = std::numeric_limits<double>::infinity();
            } else {
                pos_val = -std::numeric_limits<double>::infinity();
            }
            
        } else if (pos.is_stalemate()) {
            pos_val = 0;

        } else if (depth_left == 0 || stop || elapsed_time.count() > max_time) {
            // Evaluate if we are at eval depth or if we should finish the search
            pos_val = estimate_score(pos, own_side);

        } else {

            std::vector<std::pair<double, chess::position>> moves_scores;
            //std::vector<std::pair<double, chess::move>> moves_memes;
            int i = 0;
            for (chess::move m : pos.moves()) {
                moves_scores.push_back(std::make_pair(0, pos.copy_move(m)));
                double val = estimate_score(std::get<1>(moves_scores[i]), own_side);
                
                //moves_memes.push_back(std::make_pair(0, m));
                //moves_memes[i].first = val;
                moves_scores[i].first = val;

                i++;
            }


            if (max_node) {
                // MAX

                // Sort elements in descending eval order
                sort(moves_scores.begin(), moves_scores.end(), sort_descending);
                // sort(moves_memes.begin(), moves_memes.end(), sort_descending);

                pos_val = -std::numeric_limits<double>::infinity();
                for (const auto& move_score : moves_scores) {
                    
                    double child_val = rec_minmax(std::get<1>(move_score), false, alpha, beta, depth_left-1, own_side, pos_scores, info, stop, start_time, max_time);

                    pos_val = std::max(pos_val, child_val);

                    if (pos_val >= beta) {
                        break;
                    }

                    alpha = std::max(pos_val, alpha);
                }
            } else {
                // MIN

                // Sort elements in ascending eval order
                sort(moves_scores.begin(), moves_scores.end(), sort_ascending);
                //sort(moves_memes.begin(), moves_memes.end(), sort_ascending);

                pos_val = std::numeric_limits<double>::infinity();
                for (const auto& move_score : moves_scores) {
                    
                    double child_val = rec_minmax(std::get<1>(move_score), true, alpha, beta, depth_left-1, own_side, pos_scores, info, stop, start_time, max_time);

                    pos_val = std::min(pos_val, child_val);

                    if (pos_val <= alpha) {
                        break;
                    }

                    beta = std::min(pos_val, beta);
                }
            }
        }

        pos_scores[pos_hash] = pos_val;

        return pos_val;
    }
}

