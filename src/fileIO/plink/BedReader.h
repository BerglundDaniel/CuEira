#ifndef BEDREADER_H_
#define BEDREADER_H_

#include <../../container/HostVector.h>
#include <../data/SNP.h>

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
  Configuration& configuration;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* BEDREADER_H_ */
