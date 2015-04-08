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
#include <PersonHandler.h>
#include <SNP.h>
#include <Configuration.h>
#include <FileReaderException.h>
#include <RiskAllele.h>
#include <Phenotype.h>
#include <SNPVectorFactory.h>
#include <AlleleStatisticsFactory.h>
#include <AlleleStatistics.h>
#include <HostVector.h>

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
  explicit BedReader(const Configuration& configuration, const Container::SNPVectorFactory* snpVectorFactory,
      const PersonHandler& personHandler, const int numberOfSNPs);
  virtual ~BedReader();

  virtual Container::SNPVector* readSNP(SNP& snp);

protected:
  explicit BedReader(const Configuration& configuration, const PersonHandler& personHandler); //Used by the mock

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

  const Configuration& configuration;
  const Container::SNPVectorFactory* snpVectorFactory;
  const PersonHandler& personHandler;
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
