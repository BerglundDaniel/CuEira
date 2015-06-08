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
 */__global__ void VectorCopyIndexes(const float* indexes, const PRECISION* vector1, PRECISION* vector2, const int length){
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;

  if(threadId < length){
    vector2[threadId] = vector1[(int)indexes[threadId]]; //FIXME
  } /* if threadId < length */
}

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* VECTORCOPYINDEXES_H_ */
