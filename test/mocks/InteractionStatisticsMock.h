#ifndef INTERACTIONSTATISTICSMOCK_H_
#define INTERACTIONSTATISTICSMOCK_H_

#include <gmock/gmock.h>
#include <ostream>

#include <InteractionStatistics.h>

namespace CuEira {

class InteractionStatisticsMock: public InteractionStatistics {
public:
  InteractionStatisticsMock() :
      InteractionStatistics() {

  }

  virtual ~InteractionStatisticsMock() {

  }

  MOCK_CONST_METHOD1(toOstream, void(std::ostream& os));

  MOCK_CONST_METHOD0(getReri, double());
  MOCK_CONST_METHOD0(getAp, double());
};

} /* namespace CuEira */

#endif /* INTERACTIONSTATISTICSMOCK_H_ */
