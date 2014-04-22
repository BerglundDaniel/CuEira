#ifndef BIMREADER_H_
#define BIMREADER_H_

#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <boost/algorithm/string.hpp>

#include "../../data/SNP.h"
#include "../../programOptions/Configuration.h"

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class BimReader {
public:
  explicit BimReader(Configuration& configuration);
  virtual ~BimReader();

  int getNumberOfSNPs();

private:
  Configuration& configuration;
  int numberOfSNPs;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* BIMREADER_H_ */
