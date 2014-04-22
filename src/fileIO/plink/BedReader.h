#ifndef BEDREADER_H_
#define BEDREADER_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <HostVector.h>
#include <Person.h>
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
class BedReader {
public:
  explicit BedReader(Configuration& configuration);
  virtual ~BedReader();

  Container::HostVector readSNP(SNP& snp);

private:
  enum Mode {
    SNPMAJOR, INDIVIDUALMAJOR
  };

  Mode mode;
  std::string bedFileStr;
  std::ifstream bedFile;
  Configuration& configuration;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* BEDREADER_H_ */
