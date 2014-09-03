#ifndef SNPVECTORFACTORYMOCK_H_
#define SNPVECTORFACTORYMOCK_H_

#include <vector>
#include <gmock/gmock.h>

#include <SNPVector.h>
#include <SNPVectorFactory.h>
#include <SNP.h>
#include <GeneticModel.h>

namespace CuEira {
namespace Container {

class SNPVectorFactoryMock: public SNPVectorFactory {
public:
  SNPVectorFactoryMock(const Configuration& configuration) :
      SNPVectorFactory(configuration) {

  }

  virtual ~SNPVectorFactoryMock() {

  }

  MOCK_CONST_METHOD4(constructSNPVector, SNPVector*(SNP&, std::vector<int>*, std::vector<int>*,
          bool));
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* SNPVECTORFACTORYMOCK_H_ */
