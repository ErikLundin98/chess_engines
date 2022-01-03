#include <random>
#include <vector>
#include <atomic>

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
    std::random_device random;
    std::mt19937 generator;
};
