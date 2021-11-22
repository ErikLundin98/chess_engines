#include <random>
#include <vector>
#include <atomic>

#include <chess/chess.hpp>
#include <uci/uci.hpp>


class engine: public uci::engine
{
public:
	engine();
	~engine() = default;

	std::string name() const override
	{
		return "NNUE engine";
	}

	std::string author() const override
	{
		return "Tripp Trapp Trull";
	}

    void setup(const chess::position& position, const std::vector<chess::move>& moves) override;
    uci::search_result search(const uci::search_limit& limit, uci::search_info& info, const std::atomic_bool& ponder, const std::atomic_bool& stop) override;
    void reset() override;

private:
	chess::position root;
    std::random_device random;
    std::mt19937 generator;
};
