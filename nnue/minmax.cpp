#include "minmax.hpp"

namespace minmax {
    const double value_map[] = {1.0, 5.0, 3.0, 3.0, 9.0, 0.0};
    double estimate_score(const chess::position &p, const chess::side own_side) 
    {

        const chess::board& b = p.get_board();
        
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


    bool sort_pos_ascending(const std::pair<double, chess::position>& p1, const std::pair<double, chess::position>& p2)
    {
        return p1.first < p2.first;
    }

    bool sort_pos_descending(const std::pair<double, chess::position>& p1, const std::pair<double, chess::position>& p2)
    {
        return p1.first > p2.first;
    }


    bool sort_move_descending(const std::pair<double, chess::move>& p1, const std::pair<double, chess::move>& p2) 
    {
        return p1.first > p2.first;
    }


    double rec_minmax(const chess::position& pos, const chess::position& root, bool max_node, double alpha, double beta, int curr_depth, int max_depth, const chess::side own_side, std::unordered_map<size_t, double>& pos_scores, std::unordered_map<size_t, double>& prev_scores, uci::search_info& info, const std::atomic_bool& stop, const std::chrono::steady_clock::time_point& start_time, float max_time) 
    {
        size_t pos_hash = pos.hash();
        bool first_search = pos_hash == root.hash();

        double pos_val;

        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_time = current_time - start_time;

        if (pos_scores.count(pos_hash) >= 1) {             
            // Already in transposition table 
            pos_val = pos_scores[pos_hash];

        } else if (pos.is_checkmate()) {
            if (pos.get_turn() == own_side) {
                info.mate(pos.fullmove() - root.fullmove());

                pos_val = std::numeric_limits<double>::infinity();
            } else {
                pos_val = -std::numeric_limits<double>::infinity();
            }
            
        } else if (pos.is_stalemate()) {
            pos_val = 0;

        } else if (curr_depth == max_depth) {
            // Evaluate if we are at eval depth
            pos_val = estimate_score(pos, own_side);
        } else if (stop || elapsed_time.count() > max_time) {
            // Stop search
            if (prev_scores.count(pos_hash) >= 1) {
                pos_val = prev_scores[pos_hash];
            } else {
                pos_val = estimate_score(pos, own_side);
            }
        } else {

            std::vector<std::pair<double, chess::position>> n_pos_scores;
            std::vector<std::pair<double, chess::move>> n_moves_scores;
            int i = 0;
            for (chess::move m : pos.moves()) {
                n_pos_scores.push_back(std::make_pair(0, pos.copy_move(m)));

                // Use estimate from previous search if there is any
                size_t n_hash = std::get<1>(n_pos_scores[i]).hash();
                if (prev_scores.count(n_hash) >= 1) {
                    n_pos_scores[i].first = prev_scores[n_hash];
                } else {
                    n_pos_scores[i].first = estimate_score(std::get<1>(n_pos_scores[i]), own_side);
                }
                

                if (first_search) {
                    n_moves_scores.push_back(std::make_pair(n_pos_scores[i].first, m));
                }

                i++;
            }


            if (max_node) {
                // MAX

                // Sort elements in descending eval order
                sort(n_pos_scores.begin(), n_pos_scores.end(), sort_pos_descending);

                if (first_search) {
                    sort(n_moves_scores.begin(), n_moves_scores.end(), sort_move_descending);
                }

                pos_val = -std::numeric_limits<double>::infinity();

                int move_i = 0;
                for (const auto& score_pos : n_pos_scores) {

                    if (first_search) {
                        info.move(std::get<1>(n_moves_scores[move_i]), move_i);
                    }
                    
                    double child_val = rec_minmax(std::get<1>(score_pos), root, false, alpha, beta, curr_depth+1, max_depth, own_side, pos_scores, prev_scores, info, stop, start_time, max_time);

                    pos_val = std::max(pos_val, child_val);

                    if (pos_val >= beta) {
                        break;
                    }

                    alpha = std::max(pos_val, alpha);

                    move_i++;
                }
            } else {
                // MIN

                // Sort elements in ascending eval order
                sort(n_pos_scores.begin(), n_pos_scores.end(), sort_pos_ascending);

                pos_val = std::numeric_limits<double>::infinity();
                for (const auto& score_pos : n_pos_scores) {
                    
                    double child_val = rec_minmax(std::get<1>(score_pos), root, true, alpha, beta, curr_depth+1, max_depth, own_side, pos_scores, prev_scores, info, stop, start_time, max_time);

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
