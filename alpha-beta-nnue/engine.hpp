#include <random>
#include <vector>
#include <atomic>
#include <chrono>
#include <utility>

#include <chess/chess.hpp>
#include <uci/uci.hpp>
#include "NNUE.hpp"


class alpha_beta_engine: public uci::engine
{
public:
	alpha_beta_engine();
	~alpha_beta_engine() = default;

	std::string name() const override
	{
		return "NNUE";
	}

	std::string author() const override
	{
		return "";
	}

	static constexpr int key_size = 24;
	static constexpr int table_size = 1 << 24;
	static constexpr int key_mask = (1 << 24) - 1;

    void setup(const chess::position& position, const std::vector<chess::move>& moves) override;
    uci::search_result search(const uci::search_limit& limit, uci::search_info& info, const std::atomic_bool& ponder, const std::atomic_bool& stop) override; //Main function
    void reset() override;

	chess::move alpha_beta_search(chess::position state, int max_depth, double* states_evaluated, double* prev_evaluated,
									uci::search_info& info, const std::atomic_bool& stop, const std::chrono::steady_clock::time_point& start_time, 
									const float max_time);
	
	double alpha_beta(chess::position& state, chess::side own_side, int depth, int max_depth, int max_depth_quiescence, double alpha, double beta, bool max_player,
						double* states_evaluated, double* prev_evaluated, uci::search_info& info,
						const std::atomic_bool& stop, const std::chrono::steady_clock::time_point& start_time, const float max_time, 
						const NNUE::accumulator& accumulator);


	double alpha_beta_quiescence(chess::position& state, chess::side own_side, int depth, int max_depth_quiescence, double alpha, double beta, bool max_player,
						double* states_evaluated, double* prev_evaluated, uci::search_info& info,
						const std::atomic_bool& stop, const std::chrono::steady_clock::time_point& start_time, const float max_time,
						const NNUE::accumulator& accumulator);

	void child_state_evals(chess::position& state, chess::side own_side, double* prev_evaluated, bool quiescence_search,
							std::vector<std::pair<chess::move, double>>& output, const NNUE::accumulator& accumulator);



private:
	chess::position root;
	NNUE::evaluator evaluator;

    static constexpr double inf = std::numeric_limits<double>::infinity();

	double evaluate(const NNUE::accumulator& accumulator, chess::side own_side);
    double old_evaluate(const chess::position& state, chess::side own_side);
    bool is_terminal(const chess::position& state);
	bool is_stable(const chess::position& state);
	bool is_quiet(const chess::position& state, const chess::move& move);

	void set_accumulator(NNUE::accumulator& new_acc, chess::move move, 
						chess::position& old_pos);

};

bool sort_ascending(const std::pair<chess::move, double>& p1, const std::pair<chess::move, double>& p2);
bool sort_descending(const std::pair<chess::move, double>& p1, const std::pair<chess::move, double>& p2);

