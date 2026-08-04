#include "runtime_info.h"
#include <iostream>
int main(){const std::string json=synthetic_cpu_test_json(false);std::cout<<json<<"\n";return json.find("\"passed\":true")!=std::string::npos&&json.find("\"finite\":true")!=std::string::npos&&json.find("\"repeat_consistent\":true")!=std::string::npos&&json.find("\"android_device_executed\":false")!=std::string::npos?0:1;}
