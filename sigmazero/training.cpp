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
#include <functional>
#include <vector>
#include <queue>
#include <iomanip>

#include <chess/chess.hpp>
#include <torch/torch.h>

#include "sigmanet.hpp"
#include "rules.hpp"
#include "sync_queue.hpp"
#include "base64.hpp"


struct replay_position
{
	torch::Tensor image, value, policy; 
};


static torch::Tensor decode(const std::string& data)
{
    torch::Tensor tensor;
    std::istringstream stream(base64_decode(data));
    torch::load(tensor, stream);
    return tensor;
}


static void replay_receiver(std::istream& stream, sync_queue<replay_position>& queue)
{
	std::string encoded_replay;
	while(std::getline(stream, encoded_replay))
	{
		try
		{
			std::string encoded_image, encoded_value, encoded_policy;
			std::istringstream(encoded_replay) >> encoded_image >> encoded_value >> encoded_policy;
			replay_position replay{decode(encoded_image), decode(encoded_value), decode(encoded_policy)};
		
			queue.push(replay);
		}
		catch(const std::exception& e)
		{
			std::cerr << "exception raised when receiving replay, ignoring it..." << std::endl;
		}
	}

	std::cerr << "one replay stream was closed unexpectedly" << std::endl;
}


int main(int argc, char** argv)
{
	if(argc < 2)
	{
		std::cerr << "missing model path" << std::endl;
		return 1;
	}
	else
	{
		std::cerr << "using model path " << argv[1] << std::endl;
	}

	// setup initial model
	std::filesystem::path model_path(argv[1]);
	
	sigmanet model = make_network();

	if(std::filesystem::exists(model_path))
	{
		torch::load(model, model_path);
		std::cerr << "loaded existing model" << std::endl;
	}
	else
	{
		torch::save(model, model_path);
		std::cerr << "saved initial model" << std::endl;
	}
	
	// receive selfplay replays
	std::vector<std::ifstream> replay_files(argv+2, argv+argc);
	sync_queue<replay_position> replay_queue;
	std::vector<std::reference_wrapper<std::istream>> replay_streams(replay_files.begin(), replay_files.end());
	std::vector<std::thread> replay_threads;

	if(replay_streams.empty())
	{
		// fall back to stdin
		replay_streams.push_back(std::cin);
	}

	std::cerr << "reading replays from " << replay_streams.size() << " streams" << std::endl;

	for(std::istream& replay_stream: replay_streams)
	{
		// one thread per stream is ok since they will mostly be blocked
		replay_threads.emplace_back(replay_receiver, std::ref(replay_stream), std::ref(replay_queue));
	}

	// check cuda support
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

	model->train();
	model->to(device);
	
	//torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.2).momentum(0.9).weight_decay(0.0001)); // varying lr

	torch::optim::Adam optimizer(model->parameters());// torch::optim::AdamOptions(0.2));

	// statistics
	unsigned long long received = 0;
	unsigned long long consumed = 0;

	// replay window
	const std::size_t window_size = 1 << 14;
	const std::size_t batch_size = 512;

	torch::Tensor window_images;
	torch::Tensor window_values;
	torch::Tensor window_policies;

	const unsigned epoch_batches = 512;		// save after this number of batches
	const unsigned checkpoint_epochs = 64;	// checkpoint after this number of saves

	unsigned batches_since_epoch = 0;
	unsigned epochs_since_checkpoint = 0;
	float epoch_running_loss = 0.0f;

	bool first_replay = true;

	// start training
	std::cerr << "starting training" << std::endl;

	while(true)
	{
		int shifted = 0;
		int discarded = 0;

		std::size_t incoming_replays = replay_queue.size();
		
		std::vector<torch::Tensor> replay_images;
		std::vector<torch::Tensor> replay_values;
		std::vector<torch::Tensor> replay_policies;

		replay_images.reserve(incoming_replays);
		replay_values.reserve(incoming_replays);
		replay_policies.reserve(incoming_replays);

		while(replay_queue.size())
		{
			while(replay_queue.size() > window_size/4)
			{
				// congested replay queue
				replay_queue.pop();
				discarded++;
			}

			replay_position replay = replay_queue.pop();

			replay_images.push_back(replay.image);
			replay_values.push_back(replay.value);
			replay_policies.push_back(replay.policy);

			received++;
			shifted++;
		}

		if(shifted > 0)
		{
			std::cerr << "window: " << shifted << " shifted, " << discarded << " discarded, " << received << " received, " << consumed << " consumed" << std::endl;
		
			if(first_replay)
			{
				first_replay = false;

				window_images = torch::stack(replay_images);
				window_values = torch::stack(replay_values);
				window_policies = torch::stack(replay_policies);
			}
			else
			{
				window_images = torch::cat({window_images, torch::stack(replay_images)});
				window_values = torch::cat({window_values, torch::stack(replay_values)});
				window_policies = torch::cat({window_policies, torch::stack(replay_policies)});
			}
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

		torch::Tensor batch_images = window_images.index({batch_sample}).to(device);
		torch::Tensor batch_values = window_values.index({batch_sample}).to(device);
		torch::Tensor batch_policies = window_policies.index({batch_sample}).to(device);

		//std::cerr << "batch ready" << std::endl;
		// train on batch
		model->zero_grad();
		auto [value, policy] = model->forward(batch_images);
		//std::cerr << "distribution label: " << batch_policies << std::endl;
		auto loss = sigma_loss(value, batch_values, policy, batch_policies);
		loss.backward();
		optimizer.step();

		consumed += batch_size;
		epoch_running_loss += loss.item<float>();

		// update model
		if(++batches_since_epoch == epoch_batches)
		{
			std::cerr << "epoch: " << epoch_running_loss/epoch_batches << " average loss, " << epoch_running_loss << " running loss, " << epochs_since_checkpoint << "/" << checkpoint_epochs << " to checkpoint" << std::endl;
			
			epoch_running_loss = 0.0f;
			batches_since_epoch = 0;

			torch::save(model, model_path);
			std::cerr << "saved model " << model_path << std::endl;

			if(++epochs_since_checkpoint == checkpoint_epochs)
			{
				epochs_since_checkpoint = 0;

				auto now = std::chrono::system_clock::now();
 				const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
				std::ostringstream out;
				out << "ckpt_" << std::put_time(std::localtime(&t_c), "%FT%T") << ".pt";
				
				std::filesystem::path checkpoint_path = out.str();
				std::filesystem::copy_file(model_path, checkpoint_path);
				std::cerr << "saved checkpoint " << checkpoint_path << std::endl;
			}
		}
	}

	return 0;
}
