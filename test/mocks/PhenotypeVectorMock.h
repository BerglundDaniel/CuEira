#ifndef PHENOTYPEVECTORMOCK_H_
#define PHENOTYPEVECTORMOCK_H_

#include <gmock/gmock.h>

#include <PhenotypeVector.h>
#include <HostVector.h>
#include <MissingDataHandler.h>

namespace CuEira {
namespace Container {

template<typename Vector>
class PhenotypeVectorMock: public PhenotypeVector<Vector> {
public:
  PhenotypeVectorMock(Vector* vector = new Vector(1)) :
      PhenotypeVector(*vector), vector(vector) {

  }

  virtual ~PhenotypeVectorMock() {
    delete vector;
  }

  MOCK_CONST_METHOD0(getNumberOfIndividualsTotal, int());
  MOCK_CONST_METHOD0(getNumberOfIndividualsToInclude, int());
  MOCK_CONST_METHOD0(getPhenotypeData, const Vector&());

  MOCK_METHOD1(applyMissing, void(const MissingDataHandler<Vector>&));
  MOCK_METHOD0(applyMissing, void());

private:
  Vector* vector;

};

} /* namespace Container */
} /* namespace CuEira */

#endif /* PHENOTYPEVECTORMOCK_H_ */
