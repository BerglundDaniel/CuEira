#ifndef MISSINGDATAHANDLERMOCK_H_
#define MISSINGDATAHANDLERMOCK_H_

#include <vector>
#include <gmock/gmock.h>

#include <AlleleStatistics.h>
#include <AlleleStatisticsFactory.h>

namespace CuEira {

template<typename Vector>
class MissingDataHandlerMock: public MissingDataHandler<Vector> {
public:
  MissingDataHandlerMock() :
      MissingDataHandler(1) {

  }

  virtual ~MissingDataHandlerMock() {

  }

  MOCK_CONST_METHOD0(getNumberOfIndividualsToInclude, int());
  MOCK_CONST_METHOD0(getNumberOfIndividualsTotal, int());

  MOCK_METHOD1(setMissing, void(const std::set<int>&));

  MOCK_CONST_METHOD2(copyNonMissing, void(const Vector&, Vector&));
};

} /* namespace CuEira */

#endif /* MISSINGDATAHANDLERMOCK_H_ */
