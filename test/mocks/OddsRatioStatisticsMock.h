#ifndef ODDSRATIOSTATISTICSMOCK_H_
#define ODDSRATIOSTATISTICSMOCK_H_

#include <gmock/gmock.h>
#include <ostream>

#include <InteractionStatistics.h>

namespace CuEira {

class OddsRatioStatisticsMock: public OddsRatioStatistics {
public:
  OddsRatioStatisticsMock() :
    OddsRatioStatistics() {

  }

  virtual ~OddsRatioStatisticsMock() {

  }

  MOCK_CONST_METHOD1(toOstream, void(std::ostream& os));

  MOCK_CONST_METHOD0(getOddsRatios, const std::vector<double>&());
  MOCK_CONST_METHOD0(getOddsRatiosLow, const std::vector<double>&());
  MOCK_CONST_METHOD0(getOddsRatiosHigh, const std::vector<double>&());
};

} /* namespace CuEira */

#endif /* ODDSRATIOSTATISTICSMOCK_H_ */
