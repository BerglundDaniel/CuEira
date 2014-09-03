#ifndef BEDREADERMOCK_H_
#define BEDREADERMOCK_H_

#include <gmock/gmock.h>

#include <BedReader.h>
#include <SNP.h>
#include <SNPVector.h>
#include <SNPVectorFactory.h>
#include <AlleleStatistics.h>
#include <AlleleStatisticsFactory.h>

namespace CuEira {
namespace FileIO {

class BedReaderMock: public BedReader {
public:
  BedReaderMock(const Configuration& configuration, const Container::SNPVectorFactory& snpVectorFactory,
      const AlleleStatisticsFactory& alleleStatisticsFactory, const PersonHandler& personHandler) :
      BedReader(configuration, snpVectorFactory, alleleStatisticsFactory, personHandler) {

  }

  virtual ~BedReaderMock() {

  }

  MOCK_CONST_METHOD1(readSNP, std::pair<const AlleleStatistics*, Container::SNPVector*>*(SNP&));

};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* BEDREADERMOCK_H_ */
