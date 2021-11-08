#include <iostream>
#include <chess/chess.hpp>
#include <chess/position.hpp>
#include <chess/board.hpp>
#include "sigmanet.hpp"

void save_model(sigmanet& net, const std::string& path) {
    torch::serialize::OutputArchive output;
    net.save(output);
    output.save_to(path);
}

/*
Initializes sigmanet model
Trains the model using dummy data
Performs inference and asserts that output looks correct
*/
int main()
{
    std::cout << "Testing sigmanet model..." << std::endl;
    torch::Device device(torch::kCPU);
    // Check cuda support
    if(torch::cuda::is_available())
    {
        device = torch::Device(torch::kCUDA);
        std::cout << "Using CUDA" << std::endl;
    }
    else {
        std::cout << "Using CPU" << std::endl;
    }

    int in_channels = 64;
    int n_blocks = 10;
    double c = 0.0001; // L2 Regularization
    // Create dummy input data
    long batch_size = 64;
    int history = 2;
    int n_moves = 8 * 8 * 73; // Change in sigmanet as well

    std::cout << "Initializing model with parameters" <<
    "#input channels" << in_channels << std::endl <<
    "#residual blocks" << n_blocks << std::endl <<
    "history" << history << std::endl;

    torch::Tensor input_state = torch::rand({batch_size, in_channels, 8, 8}, device);
    // Create dummy output data
    torch::Tensor value_targets = torch::rand({batch_size, 1}, device);
    torch::Tensor policy_targets = torch::rand({batch_size, n_moves}, device);

    std::cout << "randomised data" << std::endl;

    // Initialize model loss and optimizer
    sigmanet model(history, in_channels, n_blocks);

    std::cout << "initialised model" << std::endl;

    model.to(device);
    model.train();
    // TODO: Move model to gpu if available?

    std::cout << "Training" << std::endl;
    torch::optim::SGD optimizer(model.parameters(), 
    torch::optim::SGDOptions(0.01).momentum(0.9).weight_decay(c));
    auto loss_fn = sigma_loss; // Defined in model.h

    // Training!
    int n_epochs = 2;

    for(int i = 0 ; i < n_epochs ; i++)
    {

        std::cout << "Epoch " << i << std::endl;

        model.zero_grad();
        auto[value, policy] = model.forward(input_state);
        auto loss = loss_fn(value, value_targets, policy, policy_targets);
        std::cout << "epoch " << i << "loss: " << loss.item<float>() << std::endl;
        loss.backward();
        optimizer.step();
    }
    std::cout << "Testing" << std::endl;
    // Inference
    model.eval();
    torch::Tensor test_state = torch::rand({1, in_channels, 8, 8}, device);
    auto[value, policy] = model.forward(test_state);

    std::cout << "inference result: " << std::endl << "value: " << value << std::endl << "policy: " << policy << std::endl;

    save_model(model, "test.pt");

    std::cout << "saved model" << std::endl;

    std::cout << "testing feature conversions" << std::endl;
    chess::position pos_to_encode = chess::position::from_fen("r2qkbnr/1ppp1pp1/2n5/p3p1Qp/2bPP3/2PB4/PP3PPP/RNB1K1NR w KQkq - 0 1");
    std::cout << "position:" << std::endl << pos_to_encode.get_board().to_string() << std::endl;


}