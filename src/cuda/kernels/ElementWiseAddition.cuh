#ifndef ELEMETWISEADDITION_H_
#define ELEMETWISEADDITION_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>

namespace CuEira {
namespace CUDA {
namespace Kernel {

//From http://cpplove.blogspot.com/2012/07/a-generic-loop-unroller-based-on.html
template<size_t N> struct uint_ {
};

template<size_t N, typename Lambda, typename IterT>
__device__ inline void unroller(const Lambda& f, const IterT& iter, uint_<N>){
  unroller(f, iter, uint_<N - 1>());
  f(iter + N);
}

template<typename Lambda, typename IterT>
__device__ inline void unroller(const Lambda& f, const IterT& iter, uint_<0>){
  f(iter);
}

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */__global__ void ElementWiseAddition(const PRECISION* vector1, const PRECISION* vector2, PRECISION* result,
    const int length){
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;

  if(threadId < length){
    //result[threadId] = vector1[threadId] + vector2[threadId];
    unroller([&](const int& i){result[i] = vector1[i] + vector2[i];}, threadId, uint_<GPU_UNROLL - 1>());
  } /* if threadId < numberOfPredictors */
}

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* ELEMETWISEADDITION_H_ */
