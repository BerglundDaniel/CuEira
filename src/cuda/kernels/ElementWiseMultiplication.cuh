#ifndef ELEMENTWISEMULTIPLICATION_H_
#define ELEMENTWISEMULTIPLICATION_H_

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
 */__global__ void ElementWiseMultiplication(const PRECISION* vector1, const PRECISION* vector2, PRECISION* result, const int length) {
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;

  if(threadId < length){
    result[threadId] = vector1[threadId] * vector2[threadId];
  } /* if threadId < numberOfPredictors */
}

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* ELEMENTWISEMULTIPLICATION_H_ */
