#include "CudaSNPVector.h"

namespace CuEira {
namespace Container {
namespace CUDA {

CudaSNPVector::CudaSNPVector(SNP& snp, GeneticModel geneticModel, const DeviceVector* snpOrgExMissing,
    const std::set<int>* snpMissingData, const Stream& stream) :
    SNPVector<DeviceVector>(snp, geneticModel, snpOrgExMissing, snpMissingData), stream(stream){

}

CudaSNPVector::~CudaSNPVector(){

}

void CudaSNPVector::doRecode(int snpToRisk[3]){
  Kernel::applyGeneticModel(stream, snpToRisk, snpOrgExMissing, snpRecodedExMissing); //recode=snpToRisk[org]
}

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */
