#include "KernelWrapper.h"
#include <LogisticTransform.cuh>

namespace CuEira {
namespace CUDA {

KernelWrapper::KernelWrapper(const cudaStream_t& cudaStream) :
    cudaStream(cudaStream) {

}

KernelWrapper::~KernelWrapper() {

}

void KernelWrapper::logisticTransform(const DeviceVector& logitVector, DeviceVector& probabilites) const {
  Kernel::LogisticTransform<<<1, 5, 0, cudaStream>>>(logitVector.getMemoryPointer(), probabilites.getMemoryPointer());
}

} /* namespace CUDA */
} /* namespace CuEira */
