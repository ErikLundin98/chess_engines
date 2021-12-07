#include <random>
#include <vector>
#include <atomic>
#include <chrono>
#include <optional>

#include <sstream>

#include <chess/chess.hpp>
#include <uci/uci.hpp>

#include "engine.hpp"
#include "eval.hpp"

alpha_beta_engine::alpha_beta_engine() : root() {
	
    // most clients require these options
	opt.add<uci::option_spin>("MultiPV", 1, 1, 1);
	opt.add<uci::option_spin>("Move Overhead", 0, 0, 1);
	opt.add<uci::option_spin>("Threads", 1, 1, 1);
	opt.add<uci::option_spin>("Hash", 1, 1, 1);

	// demo options
	//opt.add<uci::option_check>("Demo Check", true);
	//opt.add<uci::option_spin>("Demo Spin", 0, -10, 10);
	//opt.add<uci::option_combo>("Demo Combo", "Apples", std::initializer_list<std::string>{"Apples", "Oranges", "Bananas"});
	//opt.add<uci::option_string>("Demo String", "Tjenare");
	//opt.add<uci::option_button>("Demo Button", []()
	//{
	//	std::cerr << "button pressed" << std::endl;
	//});
}


void alpha_beta_engine::setup(const chess::position& position, const std::vector<chess::move>& moves) {
	root = position;

	for(const chess::move& move: moves)
	{
		root.make_move(move);
	}
}


uci::search_result alpha_beta_engine::search(const uci::search_limit& limit, uci::search_info& info, const std::atomic_bool& ponder, const std::atomic_bool& stop) {
    
    info.message("search started");

    std::cout << root.to_fen() << std::endl; 

	// UCI setup
	chess::side side = root.get_turn();
	std::vector<chess::move> moves = root.moves();
	std::vector<chess::move> best_line;
	float max_time = std::min(limit.time, limit.clocks[side] / 100); // estimate ~100 moves per game
	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    double value;
    chess::move best_move;

	chess::side own_side = side; // This should change if we ponder, maybe extracted from UCI settings somehow?

    std::unordered_map<size_t, double> prev_evaluated;

    for (int eval_depth = 3; eval_depth < 4; eval_depth++) {

        std::unordered_map<size_t, double> states_evaluated;

        best_move = alpha_beta_search(root, eval_depth, states_evaluated, prev_evaluated, info, stop, start_time, max_time);

        info.depth(eval_depth);
        info.nodes(states_evaluated.size());

        prev_evaluated = states_evaluated;

        // Check if enough time has passed
        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_time = current_time - start_time;
		if (stop || elapsed_time.count() > max_time) {
			break;
		}
    }

    
    // Set info
    best_line.clear();
    best_line.push_back(best_move);

    // Check if enough time has passed
    auto current_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_time = current_time - start_time;

    std::cout << "nodes evaluated = " << prev_evaluated.size() << std::endl;

    return {best_move, std::nullopt};
}

chess::move alpha_beta_engine::alpha_beta_search(chess::position state, int max_depth, std::unordered_map<size_t, double>& states_evaluated, std::unordered_map<size_t, double>& prev_evaluated, uci::search_info& info, const std::atomic_bool& stop, const std::chrono::steady_clock::time_point& start_time, const float max_time) {
    bool is_white = state.get_turn() == chess::side_white;
    chess::move best_move = chess::move();
    double best_value = is_white ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    
    for(chess::move move : state.moves()) {
        chess::position new_state = state.copy_move(move);
        double value = alpha_beta(new_state, 0, max_depth, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), !is_white, states_evaluated, prev_evaluated, info, stop, start_time, max_time);

        if((is_white && value > best_value) || (!is_white && value < best_value)) {
            best_value = value;
            best_move = move;
        }

        // Check if enough time has passed
        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_time = current_time - start_time;
		if (stop || elapsed_time.count() > max_time) {
			break;
		}
    }

    return best_move;
}

std::vector<std::pair<chess::position, double>> alpha_beta_engine::child_state_evals(const chess::position& state, std::unordered_map<size_t, double>& prev_evaluated, bool quiescence_search) {
    std::vector<std::pair<chess::position, double>> child_evals;
    for(chess::move move : state.moves()) {
        // if(quiescence_search && is_quiet(state, move))
        //     continue;

        chess::position child_state = state.copy_move(move);

        double child_value = evaluate(state);

        /*
        if (prev_evaluated.find(child_state.hash()) == prev_evaluated.end()) {
            child_value = evaluate(child_state);
        } else {
            child_value = prev_evaluated[child_state.hash()];
        }
        */

        child_evals.push_back({child_state, child_value});
    }

    return child_evals;
}


double alpha_beta_engine::alpha_beta(chess::position state, int depth, int max_depth, double alpha, double beta, bool max_player, std::unordered_map<size_t, double>& states_evaluated, 
                std::unordered_map<size_t, double>& prev_evaluated, uci::search_info& info, const std::atomic_bool& stop, const std::chrono::steady_clock::time_point& start_time, const float max_time) {    

    //if(depth >= max_depth && !is_stable(state)) {
    //    return alpha_beta_quiescence(state, 0, alpha, beta, max_player, states_evaluated);
    //}

    static int ret_counter = 0;

    size_t pos_hash = state.hash();
    
    if (depth >= max_depth || is_terminal(state)) {
        double eval = evaluate(state);
        states_evaluated[pos_hash] = eval;
        //std::cout << ++ret_counter << std::endl;
        return eval;
    }

    // if(states_evaluated.find(pos_hash) != states_evaluated.end()) {
    //     return states_evaluated[pos_hash];
    // }

    auto current_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_time = current_time - start_time;

    if (stop || elapsed_time.count() > max_time) {
        double eval = evaluate(state);
        /*
        if (prev_evaluated.find(pos_hash) == prev_evaluated.end()) {
            eval = evaluate(state);
        } else {
            eval = prev_evaluated[pos_hash];
        }*/

        states_evaluated[pos_hash] = eval;
        return eval;
    }

    std::vector<std::pair<chess::position, double>> state_evals = child_state_evals(state, prev_evaluated, false);

    double value;

    if(max_player) {
        sort(state_evals.begin(), state_evals.end(), sort_descending);

        value = -std::numeric_limits<double>::infinity();
        
        for(auto state_eval : state_evals) {
            chess::position child_state = state_eval.first;
            value = std::max(value, alpha_beta(child_state, depth + 1, max_depth, alpha, beta, false, states_evaluated, prev_evaluated, info, stop, start_time, max_time));

            if(value >= beta) {
                break;
            }
            
            alpha = std::max(alpha, value);
        }
    }
    else {
        sort(state_evals.begin(), state_evals.end(), sort_ascending);

        value = std::numeric_limits<double>::infinity();

        for(auto state_eval : state_evals) {
            chess::position child_state = state_eval.first;
            value = std::min(value, alpha_beta(child_state, depth + 1, max_depth, alpha, beta, true, states_evaluated, prev_evaluated, info, stop, start_time, max_time));

            if(value <= alpha) {
                break;
            }

            beta = std::min(beta, value);
        }
    }

    states_evaluated[pos_hash] = value;

    //std::cout << ++ret_counter << std::endl;

    return value;
}


void alpha_beta_engine::reset() {
    return;
}

// pawn, rook, knight, bishop, queen, king
const double value_map[6] = {1.0, 5.0, 3.0, 3.0, 9.0, 0.0};

double alpha_beta_engine::evaluate(const chess::position& state) {
    //return ::eval::evaluate(state);

    if(state.is_checkmate()) {
        return state.get_turn() == chess::side_white ? -inf : inf;
    }
    else if(state.is_stalemate()) {
        return 0.0;
    }
    
    else {
        double value = 0.0;

        for(int sq_int = chess::square_a1; sq_int <= chess::square_h8; sq_int++) {
            chess::square sq = static_cast<chess::square>(sq_int);
            std::pair<chess::side, chess::piece> side_piece = state.pieces().get(sq);
            chess::side side = side_piece.first;
            chess::piece piece = side_piece.second;

            if(piece == chess::piece_none)
                continue;
            
            if(side == chess::side_white)
                value += value_map[piece];
            else 
                value -= value_map[piece];
        }

        return value;
    }
}

bool alpha_beta_engine::is_terminal(const chess::position& state) {
    return state.is_checkmate() || state.is_stalemate();
}



bool sort_ascending(const std::pair<chess::position, double>& p1, const std::pair<chess::position, double>& p2) {
   return p1.second < p2.second;
}

bool sort_descending(const std::pair<chess::position, double>& p1, const std::pair<chess::position, double>& p2) {
   return p1.second > p2.second;
}









/* double alpha_beta_engine::alpha_beta(chess::position state, int max_depth, double alpha, double beta, bool max_player, std::unordered_map<size_t, double>& states_evaluated, uci::search_info& info, 
                                    const std::atomic_bool& stop, const std::chrono::steady_clock::time_point& start_time, const float max_time,
                                    std::vector<chess::move>& path) {

    auto current_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_time = current_time - start_time;
    
    // Base case -> return state evaluation
    if(max_depth == 0 || is_terminal(state)) {
        return evaluate(state);
    }

    if (stop || elapsed_time.count() > max_time) {
        return evaluate(state);
    } 

    // Return state score if it has already been calculated
    size_t pos_hash = state.hash();

    // if(states_evaluated.find(pos_hash) != states_evaluated.end()) {
    //      return states_evaluated[pos_hash];
    // }

    // Evaluate all child states
    std::vector<search_child> state_evals;
    for (chess::move move : state.moves()) {
        chess::position child_state = state.copy_move(move);
        double value;
        if(states_evaluated.count(child_state.hash()) >= 1) {
            value = states_evaluated[child_state.hash()];
        } 
        else {
            value = evaluate(child_state);
        }
        state_evals.push_back({child_state, value, move});
    }

    double value;

    if(max_player) {
        sort(state_evals.begin(), state_evals.end(), sort_descending);
        value = -inf;
        
        for(auto state_eval : state_evals) {
            chess::position child_state = state_eval.position;
            path.push_back(state_eval.move);
            value = std::max(value, alpha_beta(child_state, max_depth - 1, alpha, beta, false, states_evaluated, info, stop, start_time, max_time, path));
            
            // if(value > beta) {
            //     break;
            // }

            if(value > alpha) {
                if(max_depth == 1) {
                    info.line(path);
                }
                alpha = value;
            } 
            path.pop_back();               
        }
    }
    else {
        sort(state_evals.begin(), state_evals.end(), sort_ascending);

        value = inf;

        for(auto state_eval : state_evals) {
            chess::position child_state = state_eval.position;
            path.push_back(state_eval.move);
            value = std::min(value, alpha_beta(child_state, max_depth - 1, alpha, beta, true, states_evaluated, info, stop, start_time, max_time, path));

            // if(value < alpha) {
            //     break;
            // }
                
            if(value < beta) {
                if(max_depth == 1) {
                    info.line(path);
                }
                beta = value;
            }
            path.pop_back();

        }
    }

    states_evaluated[pos_hash] = value;

    return value;
}
*/



/*
uci::search_result alpha_beta_engine::search(const uci::search_limit& limit, uci::search_info& info, const std::atomic_bool& ponder, const std::atomic_bool& stop) {
	info.message("search started");

	// UCI setup
	chess::side side = root.get_turn();
	std::vector<chess::move> moves = root.moves();
	std::vector<chess::move> best_line;
	float max_time = std::min(limit.time, limit.clocks[side] / 100); // estimate ~100 moves per game
	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

	// Iterative deepening setup
	chess::side own_side = side; // This should change if we ponder, maybe extracted from UCI settings somehow?
	std::unordered_map<size_t, double> states_evaluated;

    double value;
    chess::move best_move;

	// Iterative deepening
	for (int eval_depth = 0; eval_depth < 8; eval_depth++) {
		// Set info
		info.depth(eval_depth);
		info.nodes(states_evaluated.size());

		// Set search datastructures
		std::unordered_map<size_t, double> pos_scores;
		    
        bool is_white = root.get_turn() == own_side; //Change latter
        best_move = chess::move();
        double best_value = is_white ? -inf : inf;
        
        int move_counter = 0;
        for(chess::move move : root.moves()) {
            info.move(move, move_counter++);
            chess::position new_state = root.copy_move(move);
            std::vector<chess::move> path{move};
            value = alpha_beta(new_state, eval_depth, -inf, inf, !is_white, states_evaluated, info, stop, start_time, max_time, path);

            if((is_white && value >= best_value) || (!is_white && value <= best_value)) {
                best_value = value;
                best_move = move;
            }
        }

        best_line.clear();
        best_line.push_back(best_move);

		// Check if enough time has passed
        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_time = current_time - start_time;
		if (stop || elapsed_time.count() > max_time) {
			break;
		}
	}

	if (best_line.empty()) {
        std::cout << "Best_line is empty" << std::endl;
		best_line.push_back(moves[0]);
	}

    //std::cout << "returning " << best_line.front().to_lan() << " eval score = " << value << std::endl;
    std::string output = "returning " + best_line.front().to_lan() + " eval score = " + std::to_string(value) + "\n";
    info.message(output);

    // if(best_line.front().to_lan() == "--") {
    //     std::cout << "found error " << root.moves()[0].to_lan() << " value = " << value << " best move = " << best_move.to_lan() << std::endl;
    // }

	return {best_line.front(), std::nullopt};
}
*/
