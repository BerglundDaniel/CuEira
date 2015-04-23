#ifndef ENVIRONMENTVECTORMOCK_H_
#define ENVIRONMENTVECTORMOCK_H_

#include <gmock/gmock.h>

#include <EnvironmentVector.h>
#include <Recode.h>
#include <EnvironmentFactor.h>
#include <EnvironmentFactorHandler.h>
#include <MissingDataHandler.h>

namespace CuEira {
namespace Container {

template<typename Vector>
class EnvironmentVectorMock: public EnvironmentVector<Vector> {
public:
  EnvironmentVectorMock(EnvironmentFactor* environmentFactor = new EnvironmentFactor(Id("env1")), Vector* vector =
      new Vector(1)) :
      EnvironmentVector(*environmentFactor, *vector), environmentFactor(environmentFactor), vector(vector) {

  }

  virtual ~EnvironmentVectorMock() {
    delete environmentFactor;
    delete vector;
  }

  MOCK_CONST_METHOD0(getEnvironmentFactor, const EnvironmentFactor&());
  MOCK_CONST_METHOD0(getNumberOfIndividualsTotal, int());
  MOCK_CONST_METHOD0(getNumberOfIndividualsToInclude, int());
  MOCK_CONST_METHOD0(getEnvironmentData, const Vector& ());

  MOCK_METHOD0(getEnvironmentData, Vector&());

  MOCK_METHOD1(recode, void(Recode));
  MOCK_METHOD2(recode, void(Recode, const MissingDataHandler<Vector>&));

protected:
  virtual void recodeProtective(){

  }

  virtual void recodeAllRisk(){

  }

  EnvironmentFactor* environmentFactor;
  Vector* vector;

};

} /* namespace Container */
} /* namespace CuEira */

#endif /* ENVIRONMENTVECTORMOCK_H_ */
