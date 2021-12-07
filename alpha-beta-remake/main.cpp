#include <iostream>
#include <memory>
#include <unordered_map>
#include <string>

#include <torch/torch.h>
#include <chess/chess.hpp>
#include <uci/uci.hpp>

#include "engine.hpp"


int main(int argc, char** argv)
{
	if(torch::cuda::is_available())
	{
		std::cout << "cuda available" << std::endl;
	}
	else
	{
		std::cout << "cuda not available" << std::endl;
	}

	chess::init();
	alpha_beta_engine engine;
	const uci::search_limit limit;
	uci::search_info info;
	const std::atomic_bool ponder{};
	const std::atomic_bool stop{};

	//engine.search(limit, info , ponder, stop);

	//std::cout << "done " << std::endl;

	return uci::main(engine);
}
