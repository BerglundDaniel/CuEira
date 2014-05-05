#ifndef BEDREADER_H_
#define BEDREADER_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <stdexcept>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

#include <HostVector.h>
#include <Person.h>
#include <PersonHandler.h>
#include <SNP.h>
#include <Configuration.h>
#include <FileReaderException.h>
#include <GeneticModel.h>
#include <RiskAllele.h>
#include <Phenotype.h>

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class BedReader {
public:
  explicit BedReader(const Configuration& configuration, const PersonHandler& personHandler, const int numberOfSNPs);
  virtual ~BedReader();

  Container::LapackppHostVector* readSNP(SNP& snp) const;

private:
  enum Mode {
    SNPMAJOR, INDIVIDUALMAJOR
  };

  /**
   * Get the bit at position in the byte, position in range 0-7
   */
  bool getBit(unsigned char byte, int position) const;
  void excludeSNP(SNP& snp) const;
  void closeBedFile(std::ifstream& bedFile) const;
  void openBedFile(std::ifstream& bedFile) const;

  const int readBufferSizeMaxSNPMAJOR = 100000; //10kb
  const int headerSize = 3;
  const int numberOfSNPs;
  Mode mode;
  const GeneticModel geneticModel;
  const std::string bedFileStr;
  const Configuration& configuration;
  const PersonHandler& personHandler;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* BEDREADER_H_ */
