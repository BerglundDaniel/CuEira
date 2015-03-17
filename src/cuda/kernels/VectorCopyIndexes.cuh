#ifndef VECTORCOPYINDEXES_H_
#define VECTORCOPYINDEXES_H_

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
 */__global__ void VectorCopyIndexes(PRECISION* vector1, const PRECISION* vector2, PRECISION* indexes,
    const int length) {
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;

  if(threadId < length){
    vector1[threadId] = vector2[indexes[threadId]];
  } /* if threadId < numberOfPredictors */
}

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* VECTORCOPYINDEXES_H_ */
