#include "CudaSNPVector.h"

namespace CuEira {
namespace Container {
namespace CUDA {

CudaSNPVector::CudaSNPVector(SNP& snp, GeneticModel geneticModel, const DeviceVector* snpOrgExMissing,
    const std::set<int>* snpMissingData, const KernelWrapper& kernelWrapper) :
    SNPVector<DeviceVector>(snp, geneticModel, snpOrgExMissing, snpMissingData), kernelWrapper(kernelWrapper) {

}

CudaSNPVector::~CudaSNPVector() {

}

void CudaSNPVector::doRecode(int snpToRisk[3]) {
  kernelwrapper.applyGeneticModel(snpToRisk, snpOrgExMissing, snpRecodedExMissing); //recode=snpToRisk[org]
}

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */
