#ifndef ALLELESTATISTICSMOCK_H_
#define ALLELESTATISTICSMOCK_H_

#include <vector>
#include <gmock/gmock.h>
#include <ostream>

#include <AlleleStatistics.h>
#include <AlleleStatisticsFactory.h>

namespace CuEira {

class AlleleStatisticsMock: public AlleleStatistics {
public:
  AlleleStatisticsMock() :
  AlleleStatistics(nullptr, nullptr){

  }

  virtual ~AlleleStatisticsMock() {

  }

  MOCK_CONST_METHOD0(getAlleleNumbers, const std::vector<int>&());
  MOCK_CONST_METHOD0(getAlleleFrequencies, const std::vector<double>&());

  MOCK_CONST_METHOD1(toOstream, void(std::ostream& os));
};

} /* namespace CuEira */

#endif /* ALLELESTATISTICSMOCK_H_ */
