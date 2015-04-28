#ifndef PHENOTYPEHANDLERMOCK_H_
#define PHENOTYPEHANDLERMOCK_H_

#include <gmock/gmock.h>
#include <string>
#include <iostream>
#include <vector>

#include <PhenotypeHandler.h>

namespace CuEira {

template<typename Vector>
class PhenotypeHandlerMock: public PhenotypeHandler<Vector> {
public:
  PhenotypeHandlerMock() :
      PhenotypeHandler(new Vector(1)) {

  }

  virtual ~PhenotypeHandlerMock() {

  }

  MOCK_CONST_METHOD0(getNumberOfIndividuals, int());
  MOCK_CONST_METHOD0(getPhenotypeData, const Vector&());

};

} /* namespace CuEira */

#endif /* PHENOTYPEHANDLERMOCK_H_ */
