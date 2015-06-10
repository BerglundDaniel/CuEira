#include "CudaSNPVectorFactory.h"

namespace CuEira {
namespace Container {
namespace CUDA {

CudaSNPVectorFactory::CudaSNPVectorFactory(const Configuration& configuration, const HostToDevice& hostToDevice,
    const KernelWrapper& kernelWrapper) :
    SNPVectorFactory(configuration), hostToDevice(hostToDevice), kernelWrapper(kernelWrapper){

}

CudaSNPVectorFactory::~CudaSNPVectorFactory(){

}

CudaSNPVector* CudaSNPVectorFactory::constructSNPVector(SNP& snp, PinnedHostVector* originalSNPData,
    const std::set<int>* snpMissingData) const{

  updateSize(originalSNPData, snpMissingData);
  DeviceVector* originalSNPDataDevice = hostToDevice.transferVector(*originalSNPData);
  delete originalSNPData;

  return new CudaSNPVector(snp, geneticModel, originalSNPDataDevice, snpMissingData, kernelWrapper);
}

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */
