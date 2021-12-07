#include <iostream>
#include <filesystem>

#include <torch/torch.h>
#include <chess/chess.hpp>
#include <sigmazero/drl/sigmanet.hpp>
#include <sigmazero/drl/action_encodings.hpp>
#include <algorithm>


int main(int argc, char** argv)
{
    int n_files = 1;
    if(argc > 1) n_files = std::stoi(argv[1]);
    std::string file_delim = "I";
    if(argc > 2) file_delim = std::stoi(argv[2]);
    int batch_size = 128;
    if(argc > 3) batch_size = std::stoi(argv[3]);
    int epochs = 1;
    if(argc > 4) epochs = std::stoi(argv[4]);
    
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

    sigmanet model(0, 256, 10);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));

    model->to(device);

    std::cout << "amount of files: " << n_files << std::endl;
    std::cout << "file_delimiter: " << file_delim << std::endl;
    std::string features_fname = "features.pt";
    std::string policies_fname = "policies.pt";
    std::string values_fname = "values.pt";

    for(int epoch = 1 ; epoch <= epochs ; ++epoch)
    {
        for(int file = 0 ; file < n_files ; ++file)
        {

            torch::Tensor features;
            torch::Tensor policies;
            torch::Tensor values;
            
            torch::load(features, std::filesystem::path(features_fname));
            torch::load(policies, std::filesystem::path(policies_fname));
            torch::load(values, std::filesystem::path(values_fname));
            std::cout << features.sizes() << policies.sizes() << values.sizes() << std::endl;

            int dataset_size = values.sizes()[0];
            int tensor_idx = 0;
            while(tensor_idx <= dataset_size)
            {
                torch::indexing::Slice batch_slice(tensor_idx, std::min(tensor_idx+batch_size, dataset_size-1));
                torch::Tensor feature_batch = features.index({batch_slice}).to(device);
                torch::Tensor policy_batch = torch::one_hot(policies.index({batch_slice}), 64*73).to(torch::kFloat).to(device);
                torch::Tensor value_batch = values.index({batch_slice}).unsqueeze(-1).to(torch::kFloat).to(device);
                std::cout << feature_batch.sizes() << policy_batch.sizes() << value_batch.sizes() << std::endl;
                model->zero_grad();
                auto[value, policy] = model->forward(feature_batch);
                auto loss = sigma_loss(policy, policy_batch, value, value_batch);
                loss.backward();
                optimizer.step();

                std::cout << 
                "epoch " << epoch <<
                "dataset progress " << tensor_idx << "/" << dataset_size <<
                "file " << file <<
                "loss " << loss.item<float>() << std::endl;
                
                tensor_idx += batch_size;
            }
            features_fname = file_delim + features_fname;
            policies_fname = file_delim + policies_fname;
            values_fname = file_delim + values_fname;
        }
        
        std::cout << "epoch " << epoch << " done. Saving model..." << std::endl;
        std::string model_path = "pretrained_model_"+std::to_string(epoch)+".pt";
        torch::save(model, std::filesystem::path(model_path));
        
    }

}