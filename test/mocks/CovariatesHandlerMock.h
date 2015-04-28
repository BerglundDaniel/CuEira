#ifndef COVARIATESHANDLERMOCK_H_
#define COVARIATESHANDLERMOCK_H_

#include <vector>
#include <gmock/gmock.h>
#include <ostream>

#include <CovariatesHandler.h>

namespace CuEira {
namespace Container {

template<typename Matrix>
class CovariatesHandlerMock: public CovariatesHandler<Matrix> {
public:
  CovariatesHandlerMock() :
      CovariatesHandler(new Matrix(1, 1)) {

  }

  virtual ~CovariatesHandlerMock() {

  }

  MOCK_CONST_METHOD0(getNumberOfCovariates, int());
  MOCK_CONST_METHOD0(getCovariatesMatrix, const Matrix&());
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* COVARIATESHANDLERMOCK_H_ */
