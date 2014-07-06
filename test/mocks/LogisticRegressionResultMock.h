#ifndef LOGISICREGRESSIONRESULTMOCK_H_
#define LOGISICREGRESSIONRESULTMOCK_H_

#include <gmock/gmock.h>

#include <LogisticRegressionResult.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <Recode.h>

namespace CuEira {
namespace Model {
namespace LogisticRegression {

class LogisticRegressionResultMock: public LogisticRegressionResult {
public:
  LogisticRegressionResultMock() :
      LogisticRegressionResult(nullptr, nullptr, nullptr, 0, 0) {

  }

  virtual ~LogisticRegressionResultMock() {

  }

  MOCK_CONST_METHOD0(getBeta, const Container::HostVector&());
  MOCK_CONST_METHOD0(getInformationMatrix, const Container::HostMatrix&());
  MOCK_CONST_METHOD0(getInverseInformationMatrix, const Container::HostMatrix&());
  MOCK_CONST_METHOD0(getNumberOfIterations, int());
  MOCK_CONST_METHOD0(getLogLikelihood, PRECISION());
  MOCK_CONST_METHOD0(calculateRecode, Recode());

};

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif /* LOGISICREGRESSIONRESULTMOCK_H_ */
