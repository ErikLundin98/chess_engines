#include <iostream>
#include <memory>
#include <unordered_map>
#include <string>

#include <torch/torch.h>
#include <chess/chess.hpp>
#include <uci/uci.hpp>

//#include "engine.hpp"
#include "NNUE.hpp"



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

	torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

	torch::Tensor tensor = torch::eye(3);
  	std::cerr << tensor << std::endl;

	chess::init();

	const std::string param_path = "/home/marno874/tdde19/evaluation-model/models/params/";
	NNUE::evaluator evaluator(param_path);
	NNUE::accumulator accumulator;

	std::vector<std::string> fens{
        "rn1qkbnr/pp3pp1/2p1p2p/3p1b2/2PP1B2/P3PN2/1P3PPP/RN1QKB1R b KQkq - 0 1",
        "rnbqkb1r/pp3ppp/2p2n2/3p4/8/P5P1/1P1PPPBP/RNBQK1NR w KQkq - 0 1",
        "r2qkbnr/p4ppp/2p1p3/2ppPb2/3P4/2P5/PP3PPP/RNBQK1NR w KQkq - 0 1",
        "rnbqk1nr/ppp2ppp/3b4/3p4/8/3P1NP1/PP2PP1P/RNBQKB1R b KQkq - 0 1",
        "rnbqkbnr/1p3ppp/p3p3/3p4/2pP4/3BPN2/PPP2PPP/RNBQ1RK1 w kq - 0 1",
        "r1bqkb1r/pp1p1pp1/2n2n1p/2p1p3/2P1P3/P1NP1N2/1P3PPP/R1BQKB1R b KQkq - 0 1",
        "r1bqkbnr/pp2pppp/2n5/2p5/3pP3/3P1NP1/PPP2P1P/RNBQKB1R w KQkq - 0 1",
        "rnbqkbnr/pp3ppp/4p3/3p4/3N4/1P6/PBP1PPPP/RN1QKB1R b KQkq - 0 1",
        "rnbq1rk1/1pppp1bp/p4np1/5p2/2PP1P2/2N1PN2/PP4PP/R1BQKB1R w KQ - 0 1",
        "r1bqk2r/ppp1bppp/2n2n2/3pp3/2P5/PP2P3/1B1P1PPP/RN1QKBNR w KQkq - 0 1",
        "r1bqkb1r/pp3ppp/2n2n2/2p1p3/2Pp4/3P1NP1/PP2PPBP/RNBQR1K1 b kq - 0 1",
        "r1bq1rk1/pp2ppbp/1npp1np1/8/2PP4/2N2NPP/PP2PPB1/R1BQ1RK1 w - - 0 1",
        "r2qkb1r/pp1npppp/2p2n2/3p1b2/2PP4/P1N1P2P/1P3PP1/R1BQKBNR b KQkq - 0 1",
        "r2q1rk1/pp2ppbp/2npbnp1/2p5/2P5/P1NP1NP1/1P2PPBP/R1BQ1RK1 w - - 0 1",
        "r1bqk1nr/pp3pbp/2np2p1/2p1p3/4P3/2N3PP/PPPPNPB1/R1BQ1RK1 b kq - 0 1",
        "rnbqkbnr/pp2ppp1/3p3p/2p5/4P3/3P1NP1/PPP2P1P/RNBQKB1R b KQkq - 0 1",
        "r1b1k2r/ppp1bppp/2n2n2/3qp3/8/2NP1N2/PPP1BPPP/R1BQ1RK1 b kq - 0 1",
        "r2qkbnr/pbpn1ppp/1p1pp3/8/3P4/1P2PN2/PBP2PPP/RN1QKB1R w KQkq - 0 1",
        "r1bqkb1r/pp3ppp/2n1p3/2ppPn2/3P4/2P2NPB/PP3P1P/RNBQK2R b KQkq - 0 1"
    };

	for(auto& fen : fens) {
		chess::position pos = chess::position::from_fen(fen); 
		accumulator.refresh(evaluator, NNUE::white, pos);
		accumulator.refresh(evaluator, NNUE::black, pos);

		//accumulator.print_accumulator(NNUE::white);
		//accumulator.print_accumulator(NNUE::black);

		if(pos.get_turn() == chess::side_black) {
			torch::Tensor x = torch::cat({accumulator.accumulator_black.unsqueeze(1), accumulator.accumulator_white.unsqueeze(1)}, 0);
			std::cout << "evaluation score = " << evaluator.forward(x) << std::endl;
		}
		else {
			torch::Tensor x = torch::cat({accumulator.accumulator_white.unsqueeze(1), accumulator.accumulator_black.unsqueeze(1)}, 0);
			std::cout << "evaluation score = " << evaluator.forward(x) << std::endl;
		}
	}

	



	//engine engine;
	//return uci::main(engine);
    return 0;
}
