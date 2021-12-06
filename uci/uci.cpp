#include <string>
#include <vector>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <numeric>
#include <memory>
#include <optional>
#include <stdexcept>
#include <functional>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <thread>
#include <algorithm>
#include <future>
#include <cctype>

#include <chess/chess.hpp>

#include "uci.hpp"
#include "output_thread.hpp"

namespace uci
{


option::option(option_type type):
type{type}
{}


std::string option::to_string() const
{
    std::ostringstream out;
    insert(out);
    return out.str();
}


std::ostream& operator<<(std::ostream& out, const option& option)
{
    option.insert(out);
    return out;
}




option_check::option_check(bool default_value):
option(option_type::check),
value{default_value}
{}

void option_check::set(const std::string& new_value)
{
    if(new_value == "true")
    {
        value = true;
    }
    else if(new_value == "false")
    {
        value = false;
    }
    else
    {
        throw std::invalid_argument("check option value not true or false");
    }
}

void option_check::insert(std::ostream& out) const
{
    out << "type check default " << (value ? "true" : "false");
}

const bool& option_check::ref() const
{
    return value;
}

option_check::operator bool() const
{
    return value;
}




uci::option_spin::option_spin(int default_value, int min, int max):
option(uci::option_type::spin),
value{default_value},
min{min},
max{max}
{}

void uci::option_spin::set(const std::string& new_value)
{
    int i = std::stoi(new_value);

    if(i < min || i > max)
    {
        throw std::invalid_argument("spin option value out of range");
    }

    value = i;
}

void uci::option_spin::insert(std::ostream& out) const
{
    out << "type spin default " << value << " min " << min << " max " << max;
}

const int& option_spin::ref() const
{
    return value;
}

uci::option_spin::operator int() const
{
    return value;
}


option_combo::option_combo(std::string_view default_value, std::initializer_list<std::string> alternatives):
option(option_type::combo),
value{default_value},
alternatives(alternatives)
{}

void option_combo::set(const std::string& new_value)
{
    if(!alternatives.contains(new_value))
    {
        throw std::invalid_argument("combo option value not among alternatives");
    }

    value = new_value;
}

void option_combo::insert(std::ostream& out) const
{
    out << "type combo default " << value;

    for(const std::string& var: alternatives)
    {
        out << " var " << var;
    }
}

const std::string& option_combo::ref() const
{
    return value;
}

option_combo::operator std::string() const
{
    return value;
}


option_button::option_button(std::function<void()> callback):
option(option_type::button),
callback{callback}
{}

void option_button::set(const std::string&)
{
    callback();
}

void option_button::insert(std::ostream& out) const
{
    out << "type button";
}


void options::set(const std::string& name, const std::string& value)
{
    holder.at(key(name))->set(value);
}


std::string options::to_string() const
{
    std::ostringstream out;
    out << *this;
    return out.str();
}


std::ostream& operator<<(std::ostream& out, const options& options)
{
	for(auto& [name, option]: options.holder)
	{
		out << "option name " << name << " " << *option << std::endl;
	}

    return out;
}

std::string options::key(const std::string& name) const
{
    //std::string key;
    //std::transform(name.begin(), name.end(), std::back_inserter(key), [](unsigned char c){return std::tolower(c);});
    //return key;
    return name;
}



search_info::search_info():
search_start{std::chrono::steady_clock::now()},
last_nodes{0},
last_nodes_time{std::chrono::steady_clock::now()}
{

}

void search_info::depth(int depth, std::optional<int> selective)
{ 
	std::ostringstream out;
	out << "info depth " << depth;

    if(selective)
    {
        out << " seldepth " << *selective;
    }

    push_message(out.str());
}

void search_info::nodes(unsigned long long nodes)
{ 
	std::ostringstream out;

    auto current_time = std::chrono::steady_clock::now();
    unsigned long long delta = nodes - last_nodes;
    
    std::chrono::duration<double> delta_time = current_time - last_nodes_time;
    unsigned long long nps = delta / delta_time.count();

	out << "info nodes " << nodes << " nps " << nps;

    last_nodes = nodes;
    last_nodes_time = current_time;

    push_message(out.str());
}

void search_info::line(const std::vector<chess::move>& best)
{ 
    if(best.empty()) return;

	std::ostringstream out;
	out << "info pv ";

    for(const chess::move& move: best)
    {
        out << move.to_lan() << ' ';
    }

    auto current_time = std::chrono::steady_clock::now();
    int time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - search_start).count();

    out << "time " << time;

    push_message(out.str());
}

void search_info::score(float pawn)
{ 
	std::ostringstream out;
    int centipawn = pawn*100;
	out << "info score cp " << centipawn;
    push_message(out.str());
}

void search_info::mate(int moves)
{ 
	std::ostringstream out;
	out << "info score mate " << moves;
    push_message(out.str());
}

void search_info::bounds(float lower, float upper)
{ 
	std::ostringstream out;
	out << "info score lowerbound " << lower << " upperbound " << upper;
    //push_message(out.str());
}

void search_info::move(const chess::move& move, int number)
{ 
	std::ostringstream out;
	out << "info currmove " << move.to_lan() << " currmovenumber " << number;
    push_message(out.str());
}

void search_info::message(const std::string& message)
{
	push_message("info string " + message);
}


engine::engine(): opt()
{
    opt.add<uci::option_spin>("MultiPV", 1, 1, 1);
    opt.add<uci::option_spin>("Move Overhead", 0, 0, 1);
    opt.add<uci::option_spin>("Threads", 1, 1, 1);
    opt.add<uci::option_spin>("Hash", 1, 1, 1);
}


// Read until token or end of stream.
static std::string extract_until(std::istream& in, std::string_view until = "")
{
    std::string result, dummy;
    while(in >> dummy)
    {
        if(dummy == until)
        {
            break;
        }

        result += dummy + ' ';
    }

    if(result.back() == ' ')
    {
        result = result.substr(0, result.size()-1);
    }

    return result;
}


//template<typename... Args>
//static void search(engine& engine, Args... args)
static void search(engine& engine, const search_limit& limit, search_info& info, const std::atomic_bool& ponder, const std::atomic_bool& stop)
{
    //search_result result = engine.search(std::forward<Args>(args)...);
    search_result result = engine.search(limit, info, ponder, stop);
    
    std::stringstream ss;

    ss << "bestmove " << result.best.to_lan();
    if(result.ponder)
    {
        ss << " ponder " << result.ponder.value().to_lan();
    }

    push_message(ss.str());
}


int main(engine& engine)
{
    std::atomic_bool running = true;
    std::atomic_bool stop = true;
    std::atomic_bool ponder = false;
    search_info info;

    std::thread output_thread(output_thread_main, std::ref(running));

	while(running)
    {
    	std::string line;
        std::getline(std::cin, line);
        std::istringstream in(line);
		std::string command;
		std::string dummy; // used for ignoring redundant parameters

		in >> command;

        if(command == "uci")
        {
            push_message("id name " + engine.name());
            push_message("id author " + engine.author());
        	push_message(engine.opt.to_string(), false);
            push_message("uciok");
        }
		else if(command == "isready")
		{
            push_message("readyok");
		}
		else if(command == "setoption")
		{
            in >> dummy;

			std::string name = extract_until(in, "value");
            std::string value = extract_until(in);

			engine.opt.set(name, value);
		}
        else if(command == "ucinewgame")
        {
            engine.reset();
        }
        else if(command == "position")
        {
            chess::position position = chess::position::from_fen(in);

            std::string lan;
            std::vector<chess::move> moves;

            in >> dummy;

            while(in >> lan)
            {
                chess::move move = chess::move::from_lan(lan);
                moves.push_back(move);
            }

            engine.setup(position, moves);
        }
        else if(command == "go")
        {
            search_limit limit;

            while(in >> command)
            {
                if(command == "searchmoves")
                {
                    std::string lan;
                    while(in >> lan)
                    {
                        chess::move move = chess::move::from_lan(lan);
                        limit.moves.push_back(move);
                    }
                }
                else if(command == "ponder")
                {
                    ponder = true;
                }
                else if(command == "wtime")
                {
                    std::string clock;
                    in >> clock;
                    limit.clocks[chess::side_white] = std::stoi(clock) / 1000.0f; // ms to s
                }
                else if(command == "btime")
                {
                    std::string clock;
                    in >> clock;
                    limit.clocks[chess::side_black] = std::stoi(clock) / 1000.0f; // ms to s
                }
                else if(command == "winc")
                {
                    std::string increment;
                    in >> increment;
                    limit.increments[chess::side_white] = std::stoi(increment) / 1000.0f; // ms to s
                }
                else if(command == "binc")
                {
                    std::string increment;
                    in >> increment;
                    limit.increments[chess::side_black] = std::stoi(increment) / 1000.0f; // ms to s
                }
                else if(command == "movestogo")
                {
                    std::string remaining;
                    in >> remaining;
                    limit.remaining_moves = std::stoi(remaining);
                }
                else if(command == "depth")
                {
                    std::string depth;
                    in >> depth;
                    limit.depth = std::stoi(depth);
                }
                else if(command == "nodes")
                {
                    std::string nodes;
                    in >> nodes;
                    limit.nodes = std::stoi(nodes);
                }
                else if(command == "mate")
                {
                    std::string mate;
                    in >> mate;
                    limit.mate = std::stoi(mate);
                }
                else if(command == "movetime")
                {
                    std::string time;
                    in >> time;
                    limit.time = std::stoi(time) / 1000.0f; // ms to s
                }
                else if(command == "infinite")
                {
                    limit.infinite = true;
                }
            }

            stop = false;
            info = search_info();
            std::thread(search, std::ref(engine), std::ref(limit), std::ref(info), std::ref(ponder), std::ref(stop)).detach();
            // todo: might want to give over ownership of engine to search thread
        }
        else if(command == "stop")
        {
            stop = true;
        }
        else if(command == "ponderhit")
        {
            // the move played is the last one in the position move list:
            // no need to stop search, but ponder mode should be exited.
            ponder = false;
        }
        else if(command == "quit")
        {
            running = false;
        }
	}

    // Wait for output thread
    output_thread.join();

    return 0;
}


}
