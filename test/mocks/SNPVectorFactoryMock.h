#ifndef SNPVECTORFACTORYMOCK_H_
#define SNPVECTORFACTORYMOCK_H_

#include <vector>
#include <gmock/gmock.h>

#include <SNPVector.h>
#include <SNPVectorFactory.h>
#include <SNP.h>
#include <GeneticModel.h>
#include <HostVector.h>

namespace CuEira {
namespace Container {

class SNPVectorFactoryMock: public SNPVectorFactory {
public:
  SNPVectorFactoryMock(const Configuration& configuration) :
      SNPVectorFactory(configuration) {

  }

  virtual ~SNPVectorFactoryMock() {

  }

  MOCK_CONST_METHOD3(constructSNPVector, SNPVector*(SNP& snp, const HostVector* originalSNPData, const std::set<int>* snpMissingData));
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* SNPVECTORFACTORYMOCK_H_ */
