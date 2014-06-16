#ifndef BEDREADERMOCK_H_
#define BEDREADERMOCK_H_

#include <gmock/gmock.h>

#include <BedReader.h>
#include <SNP.h>
#include <SNPVector.h>

namespace CuEira {
namespace FileIO {

class BedReaderMock: public BedReader {
public:
  BedReaderMock(const Configuration& configuration, const PersonHandler& personHandler) :
      BedReader(configuration, personHandler) {

  }

  virtual ~BedReaderMock() {

  }

  MOCK_CONST_METHOD1(readSNP, Container::SNPVector*(SNP&));

};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* BEDREADERMOCK_H_ */
