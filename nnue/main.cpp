#include <iostream>
#include <memory>
#include <unordered_map>
#include <string>

#include <torch/torch.h>
#include <chess/chess.hpp>
#include <uci/uci.hpp>

//#include "engine.hpp"
#include "NNUE.hpp"

// assumes weights are loaded from model a3_full
int test_refresh(NNUE::evaluator evaluator, NNUE::accumulator accumulator) {
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
    };

	std::vector<float> target_evaluations {19.5795,-1.70779,82.3437,33.9645,81.4329,22.4769,46.8259,78.7403,49.6796};
	std::vector<float> pred_evaluations;

	for(auto& fen : fens) {
		chess::position pos = chess::position::from_fen(fen); 
		accumulator.refresh(evaluator, NNUE::white, pos);
		accumulator.refresh(evaluator, NNUE::black, pos);

		pred_evaluations.push_back(evaluator.forward(accumulator.accumulator_white, accumulator.accumulator_black, pos.get_turn()));
	}

	float eps = 0.0001;

	for(int i = 0; i < pred_evaluations.size(); i++) {
		if( abs(pred_evaluations[i] - target_evaluations[i]) > eps) {
			std::cout << "Refresh test failed." << " Target was " << target_evaluations[i] << " Prediction was " << pred_evaluations[i] << std::endl;
			return 0;
 		}
	}

	return 1;
}

int test_update() {

	std::cout << "test update failed " << std::endl;
	return 0;
}


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

	std::cout << "Running tests" << std::endl;
	if (test_refresh(evaluator, accumulator) && test_update()) {
		std::cout << "All test passed!" << std::endl;
	}
	else {
		std::cout << "Tests failed. Program terminates." << std::endl;
	}


	

	chess::position pos = chess::position(); 

	if (pos.to_fen() == chess::position().to_fen()) {
		accumulator.refresh(evaluator, NNUE::white, pos);
		accumulator.refresh(evaluator, NNUE::black, pos);
		std::cout << "refresh start position" << std::endl;
	}

	auto move = pos.moves()[0];
	pos = pos.copy_move(move);

	for(auto& move : pos.moves()) {
		std::pair<chess::side, chess::piece> moved_piece = pos.pieces().get(move.from);
		
		if(moved_piece.second == chess::piece_king) {
			if(pos.get_turn() == chess::side_white) {
				//refresh white accumulator
				accumulator.refresh(evaluator, NNUE::white, pos);
				accumulator.update(evaluator, NNUE::black, move, pos.pieces());
			}
			else {
				//refresh black accumulator
				accumulator.refresh(evaluator, NNUE::black, pos);
				accumulator.update(evaluator, NNUE::white, move, pos.pieces());
			}
		}
		else {
			std::cout << "updating both" << std::endl;
			accumulator.update(evaluator, NNUE::white, move, pos.pieces());
			accumulator.update(evaluator, NNUE::black, move, pos.pieces());
		}
		break;
	}



	//engine engine;
	//return uci::main(engine);
    return 0;
}
