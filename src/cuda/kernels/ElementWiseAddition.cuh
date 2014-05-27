#ifndef ELEMETWISEADDITION_H_
#define ELEMETWISEADDITION_H_

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
 */__global__ void ElementWiseAddition(const PRECISION* vector1, const PRECISION* vector2, PRECISION* result) {
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;

  if(threadId < numberOfPredictorsDeviceConstant){
    result[threadId] = vector1[threadId] + vector2[threadId];
  } /* if threadId < numberOfPredictors */
}

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* ELEMETWISEADDITION_H_ */
