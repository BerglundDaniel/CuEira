#include "CpuSNPVector.h"

namespace CuEira {
namespace Container {
namespace CPU {

CpuSNPVector::CpuSNPVector(SNP& snp, GeneticModel geneticModel, const RegularHostVector* snpOrgExMissing,
    const std::set<int>* snpMissingData) :
    SNPVector<RegularHostVector>(snp, geneticModel, snpOrgExMissing, snpMissingData) {

}

CpuSNPVector::~CpuSNPVector() {

}

void CpuSNPVector::doRecode(int snpToRisk[3]) {
  //UNROLL
  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    (*snpRecodedExMissing)(i) = snpToRisk[(int)(*snpOrgExMissing)(i)]; //TODO possible int vector
  } /* for i */

}

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */
