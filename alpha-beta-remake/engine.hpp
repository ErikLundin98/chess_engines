#include <random>
#include <vector>
#include <atomic>
#include <chrono>
#include <utility>

#include <chess/chess.hpp>
#include <uci/uci.hpp>


class alpha_beta_engine: public uci::engine
{
public:
	alpha_beta_engine();
	~alpha_beta_engine() = default;

	std::string name() const override
	{
		return "Alpha Beta";
	}

	std::string author() const override
	{
		return "CEO of CashMoney Inc";
	}

    void setup(const chess::position& position, const std::vector<chess::move>& moves) override;
    uci::search_result search(const uci::search_limit& limit, uci::search_info& info, const std::atomic_bool& ponder, const std::atomic_bool& stop) override;
    void reset() override;

private:
	chess::position root;

    static constexpr double inf = std::numeric_limits<double>::infinity();

    double evaluate(const chess::position& state);
    bool is_terminal(const chess::position& state);
    double alpha_beta(chess::position state, int max_depth, double alpha, double beta, bool max_player, std::unordered_map<size_t, double>& states_evaluated, 
                        uci::search_info& info, const std::atomic_bool& stop, const std::chrono::steady_clock::time_point& start_time, const float max_time,
						std::vector<chess::move>& path);

	
};

struct search_child {
	chess::position position;
	double evaluation;
	chess::move move;
};

bool sort_ascending(const search_child& p1, const search_child& p2);
bool sort_descending(const search_child& p1, const search_child& p2);