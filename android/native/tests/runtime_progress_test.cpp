#include "runtime_progress.h"

#include <iostream>
#include <string>

int main()
{
    seededge_progress::reset("phase18d_test", "");
    seededge_progress::update("vae_decoder", "dec_out", "completed stage");
    seededge_progress::finish("completed");
    const std::string json = seededge_progress::json();
    const bool passed = json.find("\"state\":\"completed\"") != std::string::npos &&
                        json.find("\"phase\":\"vae_decoder\"") != std::string::npos &&
                        json.find("\"completed\"\":") == std::string::npos;
    std::cout << "runtime_progress_test passed=" << (passed ? "true" : "false") << '\n';
    return passed ? 0 : 1;
}
