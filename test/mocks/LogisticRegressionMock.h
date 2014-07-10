#ifndef LOGISTICREGRESSIONMOCK_H_
#define LOGISTICREGRESSIONMOCK_H_

#include <gmock/gmock.h>

#include <LogisticRegression.h>
#include <LogisticRegressionResult.h>

namespace CuEira {
namespace Model {
namespace LogisticRegression {

class LogisticRegressionMock: public LogisticRegression {
public:
  LogisticRegressionMock() :
      LogisticRegression() {

  }

  virtual ~LogisticRegressionMock() {

  }

  MOCK_METHOD0(calculate, LogisticRegressionResult*());
};

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif /* LOGISTICREGRESSIONMOCK_H_ */
