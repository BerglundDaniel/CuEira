#include "Configuration.h"

namespace CuEira {
namespace Configuration {

namespace options = boost::program_options;

Configuration::Configuration(int argc, char* argv[]) {
  // Declare the supported options
  options::options_description description("Allowed options");
  description.add_options()("help", "produce help message")("seed", value<int>(), "set compression level")("version,v",
      "print the version number");

  options::store(po::parse_command_line(argc, argv, description), optionsMap);
  options::notify(optionsMap);

  if(optionsMap.count("help")){
    std::cerr << description << "\n";
    std::exit(EXIT_SUCCESS);
  }

  if(optionsMap.count("version")){
    std::cerr << "Version" << CuEira_VERSION_MAJOR << "." << CuEira_VERSION_MINOR << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  if(optionsMap.count("compression")){
    std::cerr << "Compression level was set to " << optionsMap["compression"].as<int>() << ".\n";
  }else{
    std::cerr << "Compression level was not set.\n";
  }

}

Configuration::~Configuration() {

}

} //End of namespace
}
