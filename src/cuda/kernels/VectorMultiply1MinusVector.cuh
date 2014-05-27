#ifndef VECTORMULTIPLIY1MINUSVECTOR_H_
#define VECTORMULTIPLIY1MINUSVECTOR_H_

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
 */__global__ void VectorMultiply1MinusVector(const PRECISION* vector, PRECISION* result) {
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;

  if(threadId < numberOfRowsDeviceConstant){
    result[threadId] = vector[threadId] * (1 - vector[threadId]);
  } /* if threadId < numberOfPredictors */
}

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* VECTORMULTIPLIY1MINUSVECTOR_H_ */
