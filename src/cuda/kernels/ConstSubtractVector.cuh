#ifndef CONSTSUBTRACTVECTOR_H_
#define CONSTSUBTRACTVECTOR_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>

namespace CuEira {
namespace CUDA {
namespace Kernel {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
__global__ void ConstSubtractVector(const int c, PRECISION* vector, const int length) {
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;

  if(threadId < length){
    vector[threadId] = c - vector[threadId];
  } /* if threadId < numberOfPredictors */
}

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* CONSTSUBTRACTVECTOR_H_ */
