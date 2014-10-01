#ifndef COMBINEDRESULTSFACTORYMOCK_H_
#define COMBINEDRESULTSFACTORYMOCK_H_

#include <gmock/gmock.h>

#include <CombinedResults.h>
#include <CombinedResultsFactory.h>
#include <LogisticRegressionResult.h>
#include <Recode.h>

namespace CuEira {
namespace Model {

class CombinedResultsFactoryMock: public CombinedResultsFactory {
public:
  CombinedResultsFactoryMock() :
      CombinedResultsFactory() {

  }

  virtual ~CombinedResultsFactoryMock() {

  }

  MOCK_CONST_METHOD3(constructCombinedResults, CombinedResults*(LogisticRegression::LogisticRegressionResult*, LogisticRegression::LogisticRegressionResult*, Recode));
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* COMBINEDRESULTSFACTORYMOCK_H_ */
