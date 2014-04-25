#ifndef BEDREADER_H_
#define BEDREADER_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <stdexcept>

#include <HostVector.h>
#include <Person.h>
#include <SNP.h>
#include <Configuration.h>
#include <FileReaderException.h>
#include <GeneticModel.h>
#include <RiskAllele.h>

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class BedReader {
public:
  explicit BedReader(const Configuration& configuration, const Container::HostVector& outcomes,
      const int numberOfIndividuals, const int numberOfSNPs);
  virtual ~BedReader();

  Container::HostVector readSNP(SNP& snp) const;

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

  const int readBufferSizeMaxSNPMAJOR = 100;
  const int headerSize = 3;
  const int numberOfIndividuals;
  const int numberOfSNPs;
  Mode mode;
  const GeneticModel geneticModel;
  const std::string bedFileStr;
  const Configuration& configuration;
  const Container::HostVector& outcomes;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* BEDREADER_H_ */
