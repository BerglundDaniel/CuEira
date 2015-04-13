#include "CudaSNPVectorFactory.h"

namespace CuEira {
namespace Container {
namespace CUDA {

CudaSNPVectorFactory::CudaSNPVectorFactory(const Configuration& configuration, const HostToDevice& hostToDevice,
    const KernelWrapper& kernelWrapper) :
    SNPVectorFactory(configuration), hostToDevice(hostToDevice), kernelWrapper(kernelWrapper) {

}

CudaSNPVectorFactory::~CudaSNPVectorFactory() {

}

SNPVector<DeviceVector>* CudaSNPVectorFactory::constructSNPVector(SNP& snp, const HostVector* originalSNPData,
    const std::set<int>* snpMissingData) const {

  const int newSize = originalSNPData->getNumberOfRows() - snpMissingData->size();
  originalSNPData->updateSize(newSize);

  DeviceVector* originalSNPDataDevice = hostToDevice.transferVector(*originalSNPData);
  delete originalSNPData;

  return new CudaSNPVector(snp, geneticModel, originalSNPDataDevice, snpMissingData, kernelWrapper);
}

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */
