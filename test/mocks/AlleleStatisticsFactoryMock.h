#ifndef ALLELESTATISTICSFACTORYMOCK_H_
#define ALLELESTATISTICSFACTORYMOCK_H_

#include <vector>
#include <gmock/gmock.h>

#include <AlleleStatistics.h>
#include <AlleleStatisticsFactory.h>

namespace CuEira {

class AlleleStatisticsFactoryMock: public AlleleStatisticsFactory {
public:
  AlleleStatisticsFactoryMock() :
      AlleleStatisticsFactory() {

  }

  virtual ~AlleleStatisticsFactoryMock() {

  }

  MOCK_CONST_METHOD1(constructAlleleStatistics, AlleleStatistics*(const std::vector<int>*));
};

} /* namespace CuEira */

#endif /* ALLELESTATISTICSFACTORYMOCK_H_ */
