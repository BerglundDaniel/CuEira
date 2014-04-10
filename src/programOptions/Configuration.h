#ifndef CMDOPTIONS_H_
#define CMDOPTIONS_H_

#include <string>
#include <iostream>
#include <boost/program_options.hpp>

namespace CuEira {
namespace Configuration {

namespace options = boost::program_options;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class Configuration {
public:
  Configuration(int argc, char* argv[]);
  virtual ~Configuration();

private:
  options::variables_map optionsMap;
};

} //End of namespace
}

#endif /* CMDOPTIONS_H_ */
