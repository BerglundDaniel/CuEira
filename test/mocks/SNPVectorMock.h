#ifndef SNPVECTORMOCK_H_
#define SNPVECTORMOCK_H_

#include <gmock/gmock.h>

#include <SNPVector.h>
#include <Recode.h>
#include <SNP.h>
#include <Id.h>
#include <GeneticModel.h>

namespace CuEira {
namespace Container {

template<typename Vector>
class SNPVectorMock: public SNPVector<Vector> {
public:
  SNPVectorMock(SNP* snp = new SNP(Id("snp1"), "a1", "a2", 0)) :
      SNPVector(*snp, DOMINANT), snp(snp) {

  }

  virtual ~SNPVectorMock() {
    delete snp;
  }

  MOCK_CONST_METHOD0(getNumberOfIndividualsToInclude, int());
  MOCK_CONST_METHOD0(getAssociatedSNP, const SNP&());
  MOCK_CONST_METHOD0(getOriginalSNPData, const Vector&());
  MOCK_CONST_METHOD0(getSNPData, const Vector&());
  MOCK_CONST_METHOD0(hasMissing, bool());
  MOCK_CONST_METHOD0(getMissing, const std::set<int>&());

  MOCK_METHOD0(getSNPData, Vector&());

  MOCK_METHOD1(recode, void(Recode));

protected:
  void doRecode(int snpToRisk[3]){

  }

  SNP* snp;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* SNPVECTORMOCK_H_ */
