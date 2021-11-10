#include <iostream>
#include <memory>
#include <unordered_map>
#include <string>

#include <torch/torch.h>
#include <chess/chess.hpp>
#include <uci/uci.hpp>

#include "alpha_beta_engine.hpp"


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

	torch::Tensor tensor = torch::eye(3);
  	std::cerr << tensor << std::endl;

	chess::init();
	alpha_beta_engine engine;
	return uci::main(engine);
}
