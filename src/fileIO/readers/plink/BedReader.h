#ifndef BEDREADER_H_
#define BEDREADER_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <stdexcept>
#include <utility>
#include <set>
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <SNPVector.h>
#include <Person.h>
#include <PersonHandlerLocked.h>
#include <SNP.h>
#include <Configuration.h>
#include <FileReaderException.h>
#include <RiskAllele.h>
#include <Phenotype.h>
#include <SNPVectorFactory.h>
#include <AlleleStatisticsFactory.h>
#include <AlleleStatistics.h>
#include <HostVector.h>

#ifndef CPU
#include <PinnedHostVector.h>
#endif

namespace CuEira {
namespace FileIO {
class BedReaderTest;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Vector>
class BedReader {
  friend BedReaderTest;
  FRIEND_TEST(BedReaderTest, ConstructorCheckMode);
public:
  explicit BedReader(const Configuration& configuration, const Container::SNPVectorFactory<Vector>* snpVectorFactory,
      const PersonHandlerLocked& personHandler, const int numberOfSNPs);
  virtual ~BedReader();

  virtual Container::SNPVector<Vector>* readSNP(SNP& snp) const;

protected:
  explicit BedReader(const Configuration& configuration, const PersonHandlerLocked& personHandler); //Used by the mock

private:
  enum Mode {
    SNPMAJOR, INDIVIDUALMAJOR
  };

  void readSNPModeSNPMajor(SNP& snp, std::set<int>& snpMissingData, Container::HostVector& snpDataOriginal) const;
  void readSNPModeIndividualMajor(SNP& snp, std::set<int>& snpMissingData, Container::HostVector& snpDataOriginal) const;

  /**
   * Get the bit at position in the byte, position in range 0-7
   */
  bool getBit(unsigned char byte, int position) const;
  void closeBedFile(std::ifstream& bedFile) const;
  void openBedFile(std::ifstream& bedFile) const;

  const Configuration& configuration;
  const Container::SNPVectorFactory<Vector>* snpVectorFactory;
  const PersonHandlerLocked& personHandler;
  Mode mode;
  const int numberOfSNPs;
  const int numberOfIndividualsTotal;
  const std::string bedFileStr;
  int numberOfBitsPerRow;
  int numberOfBytesPerRow;
  int numberOfUninterestingBitsAtEnd;
  const static int headerSize = 3;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* BEDREADER_H_ */
