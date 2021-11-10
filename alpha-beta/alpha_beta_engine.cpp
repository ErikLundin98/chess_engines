#include <random>
#include <vector>
#include <atomic>
#include <chrono>
#include <optional>

#include <sstream>

#include <chess/chess.hpp>
#include <uci/uci.hpp>

#include "alpha_beta_engine.hpp"
#include "minmax.hpp"



alpha_beta_engine::alpha_beta_engine():
root(),
random{},
generator{random()}
{
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


void alpha_beta_engine::setup(const chess::position& position, const std::vector<chess::move>& moves)
{
	root = position;

	for(const chess::move& move: moves)
	{
		root.make_move(move);
	}
}

uci::search_result alpha_beta_engine::search(const uci::search_limit& limit, uci::search_info& info, const std::atomic_bool& ponder, const std::atomic_bool& stop)
{
	info.message("search started");

	// Debug cancer
	std::string fenstring = "rnbqkbnr/pppp1ppp/8/4P3/8/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2";
	chess::position p_meme = chess::position::from_fen(fenstring);
	double eval_val = minmax::estimate_score(p_meme, root.get_turn());
	std::ostringstream strs;
	strs << eval_val;
	std::string str = strs.str();
	info.message(str);


	// UCI setup
	chess::side side = root.get_turn();
	std::vector<chess::move> moves = root.moves();
	std::vector<chess::move> best;
	float max_time = std::min(limit.time, limit.clocks[side] / 100); // estimate ~100 moves per game
	std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

	// Iterative deepening setup
	double inf = std::numeric_limits<double>::infinity();
	chess::side own_side = side; // This should change if we ponder, maybe extracted from UCI settings somehow?
	std::unordered_map<size_t, double> prev_scores;

	// Iterative deepening
	for (int eval_depth = 1; eval_depth <= 1; eval_depth++) {
		// Set info
		info.depth(eval_depth);
		info.nodes(prev_scores.size());

		// Set search datastructures
		std::unordered_map<size_t, double> pos_scores;
		best.clear();
		
		double val = minmax::rec_minmax(root, root, true, -inf, inf, 0, eval_depth, own_side, pos_scores, prev_scores, info, stop, start_time, max_time);

		// Trace back and find the best line
		chess::position curr = root;
		chess::position p;
		for (int i = 0; i < eval_depth; i++) {
			bool found = false;
			for (chess::move m : curr.moves()) {

				p = root.copy_move(m);
				size_t p_hash = p.hash();

				if (pos_scores.count(p_hash) >= 1) {
					if (pos_scores[p_hash] == val) {
						best.push_back(m);
						curr = p;

						found = true;
						break;
					}
				}
			}

			if (!found) {
				break;
			}

		}
		info.line(best);

		// Check if enough time has passed
        auto current_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_time = current_time - start_time;
		if (stop || elapsed_time.count() > max_time) {
			break;
		}

		prev_scores = pos_scores;
	}

	if (best.empty()) {
		best.push_back(moves[0]);
	}

	return {best.front(), std::nullopt};
}
	

/*

	while(!stop)
	{
		info.depth(current_depth);
		info.nodes(std::pow(moves.size(), current_depth)); // overestimate

		// search moves
		for(std::size_t i = 0; i < moves.size(); i++)
		{
			const chess::move& move = moves[i];
			info.move(move, i);

			// stop if time is up
			bool early_stop = !limit.infinite && !ponder;
			auto current_time = std::chrono::steady_clock::now();
			std::chrono::duration<double> elapsed_time = current_time - start_time;

			if(stop || (early_stop && elapsed_time.count() >= max_time))
			{
				goto done; // innan du säger något: håll käften
			}

			// update best line to random line of depth
			best.resize(current_depth);
			chess::position p = root;

			for(int j = 0; j < current_depth; j++)
			{
				// choose random move
				std::vector<chess::move> ms = p.moves();
				std::uniform_int_distribution<int> distribution(0, ms.size()-1);
				chess::move m = ms.at(distribution(generator));
				best[j] = m;
				p = p.copy_move(m);

				// was mate accidentally found?
				if(p.is_checkmate())
				{
					info.mate(p.fullmove() - root.fullmove());

					if(early_stop)
					{
						goto done;
					}
				}
			}

			info.line(best);

			// fake search time
			std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(std::pow(2, current_depth))));
		}

		// next depth reached...
		current_depth++;
	}

done: // se ovan
	return {best.front(), std::nullopt};
}

*/

void alpha_beta_engine::reset()
{
	generator.seed(generator.default_seed);
}
