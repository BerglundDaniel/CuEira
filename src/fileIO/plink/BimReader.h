#ifndef BIMREADER_H_
#define BIMREADER_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <boost/algorithm/string.hpp>

#include <Id.h>
#include <SNP.h>
#include <Configuration.h>
#include <FileReaderException.h>

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class BimReader {
public:
  explicit BimReader(const Configuration& configuration);
  virtual ~BimReader();

  int getNumberOfSNPs() const;
  std::vector<SNP*> getSNPInformation();

private:
  const Configuration& configuration;
  int numberOfSNPs;
  std::vector<SNP*> SNPInformation;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* BIMREADER_H_ */
