#include <random>
#include <vector>
#include <atomic>
#include <chrono>
#include <optional>
#include <cstring>

#include <sstream>

#include <chess/chess.hpp>
#include <uci/uci.hpp>

#include "engine.hpp"
#include "new_eval.hpp"

alpha_beta_engine::alpha_beta_engine() : root(), evaluator("/home/marno874/tdde19/evaluation-model/models/params/") {
	
    // most clients require these options
	opt.add<uci::option_spin>("MultiPV", 1, 1, 1);
	opt.add<uci::option_spin>("Move Overhead", 0, 0, 1);
	opt.add<uci::option_spin>("Threads", 1, 1, 1);
	opt.add<uci::option_spin>("Hash", 1, 1, 1);
}   


void alpha_beta_engine::setup(const chess::position& position, const std::vector<chess::move>& moves) {
	root = position;

	for(const chess::move& move: moves)
	{
		root.make_move(move);
	}
}


void alpha_beta_engine::reset() {
    return;
}


uci::search_result alpha_beta_engine::search(const uci::search_limit& limit, uci::search_info& info, const std::atomic_bool& ponder, const std::atomic_bool& stop) {
    
    info.message("search started");

	// UCI setup
	chess::side side = root.get_turn();
	std::vector<chess::move> moves = root.moves();
	std::vector<chess::move> best_line;
	float max_time = std::min(limit.time, limit.clocks[side] / 100); // estimate ~100 moves per game
	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    double value;
    chess::move best_move;
    chess::move move = chess::move();
    bool has_completed_first = false;

    double * prev_evaluated = new double[table_size]();

    for (int eval_depth = 0;; eval_depth++) {

        if(eval_depth > 0) {
            best_move = move;
            has_completed_first = true;
        }

        double * states_evaluated = new double[table_size]();

        move = alpha_beta_search(root, eval_depth, states_evaluated, prev_evaluated, info, stop, start_time, max_time);

        info.depth(eval_depth);

        std::memcpy(prev_evaluated, states_evaluated, table_size*sizeof(double));
        delete[] states_evaluated;

        // Check if enough time has passed
        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_time = current_time - start_time;
		if (stop || elapsed_time.count() > max_time) {
             
             // In case the first iteration got interrupted
            if(!has_completed_first) {
                best_move = move;
            }
			break;
		}
    }

    
    delete[] prev_evaluated;

    
    // Set info
    best_line.clear();
    best_line.push_back(best_move);

    // Check if enough time has passed
    auto current_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_time = current_time - start_time;

    return {best_move, std::nullopt};
}


chess::move alpha_beta_engine::alpha_beta_search(chess::position state, int max_depth, double* states_evaluated, double* prev_evaluated,
									uci::search_info& info, const std::atomic_bool& stop, const std::chrono::steady_clock::time_point& start_time, 
									const float max_time) {
    
    chess::move best_move = chess::move();
    double best_value = -inf;

    chess::side own_side = state.get_turn();

    int max_depth_quiescence = 2;

    NNUE::accumulator accumulator{};
    accumulator.refresh(evaluator, NNUE::white, state);
    accumulator.refresh(evaluator, NNUE::black, state);

    for(chess::move move : state.moves()) {

        NNUE::accumulator new_accumulator(accumulator);
        set_accumulator(new_accumulator, move, state);

        chess::undo undo = state.make_move(move);

        double value = alpha_beta(state, own_side, 0, max_depth, max_depth_quiescence, -inf, inf, false, states_evaluated, prev_evaluated, info, stop, start_time, max_time, accumulator);
        
        state.undo_move(move, undo);
        
        if(value >= best_value) {
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


void alpha_beta_engine::child_state_evals(chess::position& state, chess::side own_side, double* prev_evaluated, bool quiescence_search,
							std::vector<std::pair<chess::move, double>>& output, const NNUE::accumulator& accumulator) {
        
    for (chess::move move : state.moves()) {
        /*
        if (quiescence_search && is_quiet(state, move)) {
            continue;
        }
        */
        NNUE::accumulator new_accumulator(accumulator);
        set_accumulator(new_accumulator, move, state);
        
        chess::undo undo = state.make_move(move);

        double value;
        int index = state.hash() & key_mask;

        if (prev_evaluated[index] == 0) {
            value = evaluate(new_accumulator, own_side);
        } else {
            value = prev_evaluated[index];
        }
        
        output.push_back({move, value});

        state.undo_move(move, undo);
    }
}

void alpha_beta_engine::set_accumulator(NNUE::accumulator& new_acc, chess::move move, 
						                chess::position& state) {
    
    std::pair<chess::side, chess::piece> moved_piece = state.get_board().get(move.from);
    
    //King moved -> refresh accumulator on that side
    if(moved_piece.second == chess::piece_king) {

        chess::undo undo = state.make_move(move);
        if(state.get_turn() == chess::side_white) {
            new_acc.refresh(evaluator, NNUE::white, state);
            state.undo_move(move, undo);
            
            new_acc.update(evaluator, NNUE::black, move, state);
        }
        else {
            new_acc.refresh(evaluator, NNUE::black, state);
            state.undo_move(move, undo);

            new_acc.update(evaluator, NNUE::white, move, state);
        }
    }
    // No king movement -> use move to update accumulators
    else {
        new_acc.update(evaluator, NNUE::white, move, state);
        new_acc.update(evaluator, NNUE::black, move, state);
    }
}


double alpha_beta_engine::alpha_beta(chess::position& state, chess::side own_side, int depth, int max_depth, int max_depth_quiescence, double alpha, double beta, 
                        bool max_player, double* states_evaluated, double* prev_evaluated, uci::search_info& info,
						const std::atomic_bool& stop, const std::chrono::steady_clock::time_point& start_time, const float max_time,
                        const NNUE::accumulator& accumulator) { 

    size_t pos_hash = state.hash() & key_mask;

    
    // if(depth >= max_depth && !is_stable(state)) {
    //     double eval = alpha_beta_quiescence(state, own_side, 0, max_depth_quiescence, alpha, beta, max_player, states_evaluated, prev_evaluated, info, stop, 
    //                                         start_time, max_time, accumulator);
    //     states_evaluated[pos_hash] = eval;
    //     return eval;
    // }

    if (depth >= max_depth || is_terminal(state)) {
        double eval = evaluate(accumulator, own_side);
        states_evaluated[pos_hash] = eval;
        return eval;
    }

    auto current_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_time = current_time - start_time;

    if (stop || elapsed_time.count() > max_time) {
        double eval = evaluate(accumulator, own_side);
        states_evaluated[pos_hash] = eval;
        return eval;
    }

    std::vector<std::pair<chess::move, double>> state_evals;
    child_state_evals(state, own_side, prev_evaluated, false, state_evals, accumulator);

    double value;

    if(max_player) {
        sort(state_evals.begin(), state_evals.end(), sort_descending);

        value = -std::numeric_limits<double>::infinity();

        elapsed_time = std::chrono::steady_clock::now() - start_time;

        for(size_t i = 0; i < state_evals.size() && (!stop && elapsed_time.count() < max_time); i++) {
            auto state_eval = state_evals[i]; 

            NNUE::accumulator new_accumulator(accumulator);
            set_accumulator(new_accumulator, state_eval.first, state);

            chess::undo undo = state.make_move(state_eval.first);

            
            
            value = std::max(value, alpha_beta(state, own_side, depth + 1, max_depth, max_depth_quiescence, alpha, beta, false, states_evaluated, 
                            prev_evaluated, info, stop, start_time, max_time, new_accumulator));
            state.undo_move(state_eval.first, undo);

            if(value >= beta) {
                break;
            }
            
            alpha = std::max(alpha, value);

            elapsed_time = std::chrono::steady_clock::now() - start_time;
        }
    }
    else {
        sort(state_evals.begin(), state_evals.end(), sort_ascending);

        value = std::numeric_limits<double>::infinity();

        elapsed_time = std::chrono::steady_clock::now() - start_time;

        for(size_t i = 0; i < state_evals.size() && (!stop && elapsed_time.count() < max_time); i++) {
            auto state_eval = state_evals[i]; 

            NNUE::accumulator new_accumulator(accumulator);
            set_accumulator(new_accumulator, state_eval.first, state);

            chess::undo undo = state.make_move(state_eval.first);

            value = std::min(value, alpha_beta(state, own_side, depth + 1, max_depth, max_depth_quiescence, alpha, beta, true, states_evaluated, 
                        prev_evaluated, info, stop, start_time, max_time, new_accumulator));
            state.undo_move(state_eval.first, undo);

            if(value <= alpha) {
                break;
            }

            beta = std::min(beta, value);

            elapsed_time = std::chrono::steady_clock::now() - start_time;
        }
    }

    states_evaluated[pos_hash] = value;

    return value;
}

double alpha_beta_engine::alpha_beta_quiescence(chess::position& state, chess::side own_side, int depth, int max_depth_quiescence, double alpha, double beta, bool max_player,
						double* states_evaluated, double* prev_evaluated, uci::search_info& info,
						const std::atomic_bool& stop, const std::chrono::steady_clock::time_point& start_time, const float max_time,
						const NNUE::accumulator& accumulator) {    

    size_t pos_hash = state.hash() & key_mask;


    if(depth >= max_depth_quiescence || is_stable(state) || is_terminal(state)) {
        double eval = evaluate(accumulator, own_side);
        states_evaluated[pos_hash] = eval;
        return eval;
    }

    auto current_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_time = current_time - start_time;

    if (stop || elapsed_time.count() > max_time) {
        double eval = evaluate(accumulator, own_side);
        states_evaluated[pos_hash] = eval;
        return eval;
    }

    std::vector<std::pair<chess::move, double>> state_evals;
    child_state_evals(state, own_side, prev_evaluated, true, state_evals, accumulator);

    double value;

    if(max_player) {
        sort(state_evals.begin(), state_evals.end(), sort_descending);

        value = -std::numeric_limits<double>::infinity();

        elapsed_time = std::chrono::steady_clock::now() - start_time;

        for(size_t i = 0; i < state_evals.size() && (!stop && elapsed_time.count() < max_time); i++) {
            auto state_eval = state_evals[i]; 

            NNUE::accumulator new_accumulator(accumulator);
            set_accumulator(new_accumulator, state_eval.first, state);
            chess::undo undo = state.make_move(state_eval.first);
            
            value = std::max(value, alpha_beta_quiescence(state, own_side, depth + 1, max_depth_quiescence, alpha, beta, false, states_evaluated, 
                            prev_evaluated, info, stop, start_time, max_time, new_accumulator));
            state.undo_move(state_eval.first, undo);

            if(value >= beta) {
                break;
            }
            
            alpha = std::max(alpha, value);

            elapsed_time = std::chrono::steady_clock::now() - start_time;
        }
    }
    else {
        sort(state_evals.begin(), state_evals.end(), sort_ascending);

        value = std::numeric_limits<double>::infinity();

        elapsed_time = std::chrono::steady_clock::now() - start_time;

        for(size_t i = 0; i < state_evals.size() && (!stop && elapsed_time.count() < max_time); i++) {
            auto state_eval = state_evals[i]; 

            NNUE::accumulator new_accumulator(accumulator);
            set_accumulator(new_accumulator, state_eval.first, state);
            chess::undo undo = state.make_move(state_eval.first);
            
            value = std::min(value, alpha_beta_quiescence(state, own_side, depth + 1, max_depth_quiescence, alpha, beta, true, states_evaluated, 
                        prev_evaluated, info, stop, start_time, max_time, new_accumulator));
            state.undo_move(state_eval.first, undo);

            if(value <= alpha) {
                break;
            }

            beta = std::min(beta, value);

            elapsed_time = std::chrono::steady_clock::now() - start_time;
        }
    }

    states_evaluated[pos_hash] = value;

    return value;
}

bool alpha_beta_engine::is_terminal(const chess::position& state) {
    return state.is_checkmate() || state.is_stalemate();
}


/**
 * Returns true if there are no takes or promotes are possible from the given state
 */
bool alpha_beta_engine::is_stable(const chess::position& state) {
    for(chess::move move : state.moves()) {
        if(!is_quiet(state, move))
            return false;
    }

    return true;
}


/**
 * Returns true if move is not a take or promotion
 */
bool alpha_beta_engine::is_quiet(const chess::position& state, const chess::move& move) {
    chess::square sq = move.to;
    chess::piece promote = move.promote;

    return state.get_board().get(sq).second == chess::piece_none && promote == chess::piece_none;
}


bool sort_ascending(const std::pair<chess::move, double>& p1, const std::pair<chess::move, double>& p2) {
   return p1.second < p2.second;
}

bool sort_descending(const std::pair<chess::move, double>& p1, const std::pair<chess::move, double>& p2) {
   return p1.second > p2.second;
}


// pawn, rook, knight, bishop, queen, king
const double value_map[6] = {1.0, 5.0, 3.0, 3.0, 9.0, 0.0};

double alpha_beta_engine::evaluate(const NNUE::accumulator& accumulator, chess::side own_side) {
    //return old_evaluate(state, own_side);
    //return new_eval::evaluate(state, own_side);
    
    //auto current_time = std::chrono::steady_clock::now();
	
    double eval = evaluator.forward(accumulator.accumulator_white, accumulator.accumulator_black, own_side);
    
    //std::chrono::duration<double> elapsed_time = std::chrono::steady_clock::now() - current_time;
	
    //std::cout << "done execution took " << elapsed_time.count() << " seconds" << std::endl;
    
    return eval;
}

double alpha_beta_engine::old_evaluate(const chess::position& state, chess::side own_side) {
    
    if (state.is_checkmate()) {
        return state.get_turn() == own_side ? -inf : inf;
    }
    else if (state.is_stalemate()) {
        return 0.0;
    }
    
    else {
        double value = 0.0;

        for(int sq_int = chess::square_a1; sq_int <= chess::square_h8; sq_int++) {
            chess::square sq = static_cast<chess::square>(sq_int);
            std::pair<chess::side, chess::piece> side_piece = state.get_board().get(sq);
            chess::side side = side_piece.first;
            chess::piece piece = side_piece.second;

            if(piece == chess::piece_none) {
                continue;
            }
            
            if(side == own_side) {
                value += value_map[piece];
            } else { 
                value -= value_map[piece];
            }
        }

        return value;
    }
}


