#ifndef APPLYGENTICMODEL_H_
#define APPLYGENTICMODEL_H_

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
 */__global__ void ApplyGeneticModel(const int snpToRisk[3], const float* from, float* to,
    const int length){
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;

  if(threadId < length){
    to[threadId] = snpToRisk[(int)from[threadId]]; //Bad???
  } /* if threadId < length */
}

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* APPLYGENTICMODEL_H_ */
