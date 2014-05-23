#ifndef LOGLIKELIHOODPARTS_H_
#define LOGLIKELIHOODPARTS_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>

namespace CuEira {
namespace CUDA {
namespace Kernel {

//extern __constant__ int numberOfRows;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
__global__ void LogLikelihoodParts(const PRECISION* outcomes, const PRECISION* probabilites, PRECISION* result) {
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;

  if(threadId < 5){
#ifdef DOUBLEPRECISION
    result[threadId]=outcomes[threadId]*log(probabilites[threadId])+(1-outcomes[threadId])*log(1-probabilites[threadId]);
#else
    result[threadId]=outcomes[threadId]*logf(probabilites[threadId])+(1-outcomes[threadId])*logf(1-probabilites[threadId]);
#endif
  } /* if threadId < numberOfRows */
}

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* LOGLIKELIHOODPARTS_H_ */
