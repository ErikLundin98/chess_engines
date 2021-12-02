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

double alpha_beta_engine::alpha_beta(chess::position state, int max_depth, double alpha, double beta, bool max_player, std::unordered_map<size_t, double>& states_evaluated, uci::search_info& info, 
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


bool sort_ascending(const search_child& p1, const search_child& p2) {
   return p1.evaluation < p2.evaluation;
}

bool sort_descending(const search_child& p1, const search_child& p2) {
   return p1.evaluation > p2.evaluation;
}

