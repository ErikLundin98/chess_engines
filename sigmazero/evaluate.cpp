#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <filesystem>
#include <random>
#include <memory>

#include <chess/chess.hpp>
#include <torch/torch.h>
#include "drl/sigmanet.hpp"
#include "util.hpp"


int main(int argc, char **argv)
{
	if (argc != 3)
	{
		std::cerr << "missing model path or fen" << std::endl;
		return 1;
	}
	else
	{
		std::cerr << "using model path" << argv[1] << std::endl; 
	}

    std::string fen_string(argv[2]);

	chess::init();
	torch::NoGradGuard no_grad;
	std::filesystem::path model_path(argv[1]);

	// wait for initial model
	while (!std::filesystem::exists(model_path))
	{
		std::cerr << "model does not exist" << std::endl;
        return 1;
	}

	// load initial model
	sigmanet model(0, 128, 10);

	torch::Device device(torch::kCPU);

    if(torch::cuda::is_available())
    {
         device = torch::Device(torch::kCUDA);
         std::cerr << "using CUDA" << std::endl;
    }
    else
	{
         std::cerr << "using CPU" << std::endl;
    }
    
	torch::load(model, model_path);
	model->to(device);
	model->eval();
	model->zero_grad();
	std::cerr << "loaded model" << std::endl;
    
    chess::position position = chess::position::from_fen(fen_string);
    debug_position(model, device, position);

}