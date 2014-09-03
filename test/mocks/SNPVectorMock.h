#ifndef SNPVECTORMOCK_H_
#define SNPVECTORMOCK_H_

#include <gmock/gmock.h>

#include <SNPVector.h>
#include <Recode.h>
#include <SNP.h>
#include <Id.h>
#include <StatisticModel.h>
#include <GeneticModel.h>
#include <HostVector.h>

namespace CuEira {
namespace Container {

class SNPVectorMock: public SNPVector {
public:
  SNPVectorMock(SNP& snp) :
  SNPVector(snp, DOMINANT, nullptr){

  }

  virtual ~SNPVectorMock() {

  }

  MOCK_CONST_METHOD0(getNumberOfIndividualsToInclude, int());
  MOCK_CONST_METHOD0(getOrginalData, const std::vector<int>&());
  MOCK_CONST_METHOD0(getRecodedData, const Container::HostVector&());
  MOCK_CONST_METHOD0(getAssociatedSNP, const SNP&());

  MOCK_METHOD1(recode, void(Recode));
  MOCK_METHOD2(applyStatisticModel, void(StatisticModel, const Container::HostVector&));
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* SNPVECTORMOCK_H_ */
