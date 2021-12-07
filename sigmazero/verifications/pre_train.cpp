#include <iostream>
#include <filesystem>

#include <torch/torch.h>
#include <chess/chess.hpp>
#include <sigmazero/drl/sigmanet.hpp>
#include <sigmazero/drl/action_encodings.hpp>
#include <algorithm>


int main()
{
    sigmanet model(0, 256, 10);
    torch::Tensor features;
    torch::Tensor policies;
    torch::Tensor values;
    
    torch::load(features, std::filesystem::path(">features.pth"));
    torch::load(policies, std::filesystem::path(">policies.pth"));
    torch::load(values, std::filesystem::path(">values.pth"));
    std::cout << features.sizes() << policies.sizes() << values.sizes() << std::endl;

    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));
    int dataset_size = values.sizes()[0];
    int epochs = 1;
    int batch_size = 128;
    for(int epoch = 1 ; epoch <= epochs ; ++epoch)
    {
        std::cout << "new batch" << std::endl;
        int tensor_idx = 0;
        while(tensor_idx <= dataset_size)
        {
            torch::indexing::Slice batch_slice(tensor_idx, std::min(tensor_idx+batch_size, dataset_size-1));
            torch::Tensor feature_batch = features.index({batch_slice});
            torch::Tensor policy_batch = torch::one_hot(policies.index({batch_slice}), 64*73).to(torch::kFloat);
            torch::Tensor value_batch = values.index({batch_slice}).unsqueeze(-1).to(torch::kFloat);
            std::cout << feature_batch.sizes() << policy_batch.sizes() << value_batch.sizes() << std::endl;
            model->zero_grad();
            auto[value, policy] = model->forward(feature_batch);
            std::cout << "forward done, ";
            auto loss = sigma_loss(policy, policy_batch, value, value_batch);
            std::cout << "loss done, ";
            loss.backward();
            optimizer.step();

            std::cout << 
            "epoch " << epoch << std::endl <<
            "loss " << loss.item<float>() << std::endl <<
            "batch " << epoch << std::endl;
            
            tensor_idx += batch_size;
        }
    }

}