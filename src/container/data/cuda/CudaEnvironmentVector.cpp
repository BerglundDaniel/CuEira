#include "CudaEnvironmentVector.h"

namespace CuEira {
namespace Container {
namespace CUDA {

CudaEnvironmentVector::CudaEnvironmentVector(const EnvironmentFactorHandler<DeviceVector>& environmentFactorHandler,
    const Stream& stream) :
    EnvironmentVector(environmentFactorHandler), stream(stream) {

}

CudaEnvironmentVector::~CudaEnvironmentVector() {

}

void CudaEnvironmentVector::recodeProtective() {
  if(environmentFactor.getVariableType() == BINARY){
    Kernel::constSubtractVector(stream, 1, *envExMissing);
  }else{
    Kernel::constSubtractVector(stream,environmentFactor.getMax() + environmentFactor.getMin(), *envExMissing);
  }
}

void CudaEnvironmentVector::recodeAllRisk() {
  Kernel::copyVector(stream, originalData, *envExMissing);
}

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */
