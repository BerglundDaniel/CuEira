#ifndef ELEMENTWISEDIVISION_H_
#define ELEMENTWISEDIVISION_H_

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
__global__ void ElementWiseDivision(const PRECISION* numeratorVector, const PRECISION* denomitorVector, PRECISION* result, const int length) {
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;

  if(threadId < length){
    result[threadId]=numeratorVector[threadId]/denomitorVector[threadId];
  } /* if threadId < numberOfPredictors */
}

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* ELEMENTWISEDIVISION_H_ */
