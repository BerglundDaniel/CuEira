#ifndef ELEMENTWISEABSOLUTEDIFFERENCE_H_
#define ELEMENTWISEABSOLUTEDIFFERENCE_H_

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
 */__global__ void ElementWiseAbsoluteDifference(const PRECISION* vector1, const PRECISION* vector2, PRECISION* result) {
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;

  if(threadId < numberOfPredictorsDeviceConstant){
#if PRECISION == double
    result[threadId] = fabs(vector1[threadId] - vector2[threadId]);
#else
    result[threadId] = fabsf(vector1[threadId] - vector2[threadId]);
#endif
  } /* if threadId < numberOfPredictors */
}

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* ELEMENTWISEABSOLUTEDIFFERENCE_H_ */
