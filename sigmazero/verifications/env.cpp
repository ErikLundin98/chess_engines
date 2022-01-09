#include <iostream>
#include <cstdlib>

int main()
{
    if(const char* env_p = std::getenv("BATCH_SIZE"))
        std::cout << env_p << '\n';
}