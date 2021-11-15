#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <memory>
#include <sstream>
#include <random>
#include <algorithm>
#include <cstdint>
#include <utility>
#include <future>
#include <thread>

#include <chess/chess.hpp>
#include <torch/torch.h>

#include <sigmazero/drl/sigmanet.hpp>
#include "base64.hpp"


static torch::Tensor decode(const std::string& data)
{
    torch::Tensor tensor;
	std::istringstream stream(base64_decode(data));
	torch::load(tensor, stream);
	return tensor;
}


static std::string readline(std::istream& in)
{
	std::string line;
	std::getline(in, line);
	return line;
}


int main(int argc, char** argv)
{
	sigmanet model(0, 64, 50);

	if(argc != 2)
	{
		std::cerr << "missing model path" << std::endl;
		return 1;
	}

	std::filesystem::path model_path = argv[1];
	
	if(std::filesystem::exists(model_path))
	{
		torch::load(model, model_path);
		std::cerr << "loaded existing model" << std::endl;
	}
	else
	{
		torch::save(model, model_path);
		std::cerr << "saved initial model" << std::endl;
		std::cout << std::endl; // indicate that model has been updated
	}

	// start training
	std::cerr << "starting training" << std::endl;
	torch::Device device(torch::kCPU);
	if(torch::cuda::is_available())
	{
		device = torch::Device(torch::kCUDA);
		std::cout << "Moving model to GPU" << std::endl;
	}
	else {
		std::cout << "GPU not found" << std::endl;
	}
	model.to(device);
	// Initialize optimizer etc
	// TODO: Get params as args?
	torch::optim::SGD optimizer(model.parameters(), 
    torch::optim::SGDOptions(0.01).momentum(0.9).weight_decay(0.0001));
    auto loss_fn = sigma_loss; // Defined in sigmanet.h

	// replay window
	const std::size_t window_size = 1024;
	const std::size_t batch_size = 256;

	// statistics
	unsigned long long received = 0;
	unsigned long long consumed = 0;


	torch::Tensor window_images;
	torch::Tensor window_values;
	torch::Tensor window_policies;
	
	std::future<std::string> replay_future = std::async(readline, std::ref(std::cin));
	bool first_replay = true;

	// start training
	std::cerr << "starting training" << std::endl;

	while(true)
	{
		while(replay_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
		{
			std::string replay_encoding = replay_future.get();

			std::string encoded_image;
			std::string encoded_value;
			std::string encoded_policy;

			std::istringstream(replay_encoding) >> encoded_image >> encoded_value >> encoded_policy;

			torch::Tensor replay_image = decode(encoded_image).unsqueeze(0);
			torch::Tensor replay_value = decode(encoded_value).unsqueeze(0);
			torch::Tensor replay_policy = decode(encoded_policy).unsqueeze(0);

			if(first_replay)
			{
				first_replay = false;

				window_images = replay_image;
				window_values = replay_value;
				window_policies = replay_policy;
			}
			else
			{
				window_images = torch::cat({window_images, replay_image}, 0);
				window_values = torch::cat({window_values, replay_value}, 0);
				window_policies = torch::cat({window_policies, replay_policy}, 0);
			}

			std::cerr << "selfplay result received" << std::endl;
			replay_future = std::async(readline, std::ref(std::cin));
			received++;
		}

		// wait for enough games to be available
		if(received < window_size)
		{
			continue;
		}

		// remove old replays
		torch::indexing::Slice window_slice(-window_size);

		window_images = window_images.index({window_slice});
		window_values = window_values.index({window_slice});
		window_policies = window_policies.index({window_slice});

		// sample batch of replays
		torch::Tensor batch_sample = torch::randint(window_size, {batch_size}).to(torch::kLong);

		torch::Tensor batch_images = window_images.index({batch_sample});
		torch::Tensor batch_values = window_values.index({batch_sample});
		torch::Tensor batch_policies = window_policies.index({batch_sample});

		//std::cerr << "batch ready" << std::endl;

		// train on batch
		model.train();
		
		
		model.zero_grad();
		std::pair<torch::Tensor, torch::Tensor> output = model.forward(batch_images);
		auto loss = loss_fn(output.first, batch_values, output.second, batch_policies);
		loss.backward();
		optimizer.step();
		
		consumed += batch_size;
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		// TODO: Only update model if it outperforms previous?
		// update model
		torch::save(model, model_path);
		std::cout << std::endl; // indicate that model has updated

		// show statistics
		std::cerr << "received: " << received << ", consumed: " << consumed << std::endl;
	}

	return 0;
}
