#include "CudaEnvironmentVector.h"

namespace CuEira {
namespace Container {
namespace CUDA {

CudaEnvironmentVector::CudaEnvironmentVector(const EnvironmentFactorHandler<DeviceVector>& environmentFactorHandler,
    const KernelWrapper& kernelWrapper, const CublasWrapper& cublasWrapper) :
    EnvironmentVector(environmentFactorHandler), kernelWrapper(kernelWrapper), cublasWrapper(cublasWrapper) {

}

CudaEnvironmentVector::~CudaEnvironmentVector() {

}

void CudaEnvironmentVector::recodeProtective() {
  if(environmentFactor.getVariableType() == BINARY){
    kernelWrapper.constSubtractVector(1, *envExMissing);
  }else{
    kernelWrapper.constSubtractVector(environmentFactor.getMax() + environmentFactor.getMin(), *envExMissing);
  }
}

void CudaEnvironmentVector::recodeAllRisk() {
  cublasWrapper.copyVector(originalData, *envExMissing);
}

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */
