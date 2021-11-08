#ifndef MINMAX_H
#define MINMAX_H

#include <utility>
#include <chrono>

#include <chess/chess.hpp>
#include <uci/uci.hpp>

namespace minmax {

    double estimate_score(const chess::position &p, const chess::side own_side);

    bool sort_ascending(const std::pair<double, chess::position>& p1, const std::pair<double, chess::position>& p2);

    bool sort_descending(const std::pair<double, chess::position>& p1, const std::pair<double, chess::position>& p2);

    double rec_minmax(const chess::position& pos, bool max_node, double alpha, double beta, int depth_left, const chess::side own_side, std::unordered_map<size_t, double>& pos_scores, uci::search_info& info, const std::atomic_bool& stop, const std::chrono::steady_clock::time_point& start_time, float max_time);
}

#endif /* MINMAX_H */