#ifndef BEDREADER_H_
#define BEDREADER_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <stdexcept>
#include <utility>
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <SNPVector.h>
#include <Person.h>
#include <PersonHandler.h>
#include <SNP.h>
#include <Configuration.h>
#include <FileReaderException.h>
#include <RiskAllele.h>
#include <Phenotype.h>
#include <SNPVectorFactory.h>
#include <AlleleStatisticsFactory.h>
#include <AlleleStatistics.h>

namespace CuEira {
namespace FileIO {
class BedReaderTest;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class BedReader {
  friend BedReaderTest;
  FRIEND_TEST(BedReaderTest, ConstructorCheckMode);
public:
  explicit BedReader(const Configuration& configuration, const Container::SNPVectorFactory& snpVectorFactory,
      const AlleleStatisticsFactory& alleleStatisticsFactory, const PersonHandler& personHandler,
      const int numberOfSNPs);
  virtual ~BedReader();

  virtual std::pair<const AlleleStatistics*, Container::SNPVector*>* readSNP(SNP& snp);

protected:
  explicit BedReader(const Configuration& configuration, const Container::SNPVectorFactory& snpVectorFactory,
      const AlleleStatisticsFactory& alleleStatisticsFactory, const PersonHandler& personHandler); //Used by the mock

private:
  enum Mode {
    SNPMAJOR, INDIVIDUALMAJOR
  };

  /**
   * Get the bit at position in the byte, position in range 0-7
   */
  bool getBit(unsigned char byte, int position) const;
  void closeBedFile(std::ifstream& bedFile);
  void openBedFile(std::ifstream& bedFile);

  void setSNPRiskAllele(SNP& snp, const AlleleStatistics& alleleStatistics) const;
  void setSNPInclude(SNP& snp, const AlleleStatistics& alleleStatistics) const;

  const Configuration& configuration;
  const Container::SNPVectorFactory& snpVectorFactory;
  const AlleleStatisticsFactory& alleleStatisticsFactory;
  const PersonHandler& personHandler;
  Mode mode;
  const int numberOfSNPs;
  const int numberOfIndividualsToInclude;
  const int numberOfIndividualsTotal;
  const std::string bedFileStr;
  const double minorAlleleFrequencyThreshold;
  int numberOfBitsPerRow;
  int numberOfBytesPerRow;
  int numberOfUninterestingBitsAtEnd;
  const static int headerSize = 3;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* BEDREADER_H_ */
