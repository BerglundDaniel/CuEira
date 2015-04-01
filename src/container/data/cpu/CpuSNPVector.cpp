#include "CpuSNPVector.h"

namespace CuEira {
namespace Container {
namespace CPU {

CpuSNPVector::CpuSNPVector(SNP& snp, GeneticModel geneticModel, const DeviceVector* snpOrgExMissing,
    const std::set<int>* snpMissingData) :
    SNPVector<RegularHostVector>(snp, geneticModel, snpOrgExMissing, snpMissingData) {

}

CpuSNPVector::~CpuSNPVector() {

}

void CpuSNPVector::doRecode(int snpToRisk[3]) {
  //UNROLL
  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    (*snpRecodedExMissing)(i) = snpToRisk[(*snpOrgExMissing)(i)];
  } /* for i */

}

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */
