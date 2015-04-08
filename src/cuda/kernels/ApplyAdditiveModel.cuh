#ifndef APPLYADDITIVEMODEL_H_
#define APPLYADDITIVEMODEL_H_

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
 *
 */
__global__ void ApplyAdditiveModel(PRECISION* vector1, PRECISION* vector2, PRECISION* interactionVector,
    const int length){
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;

  if(threadId < length){
    interactionVector[threadId] = vector1[threadId] * vector2[threadId];
    if(interactionVector[threadId] != 0){
      vector1[threadId] = 0;
      vector2[threadId] = 0;
    }
  } /* if threadId < length */
}

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* APPLYADDITIVEMODEL_H_ */
