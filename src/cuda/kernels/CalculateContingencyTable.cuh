#ifndef CALCULATECONTINGENCYTABLE_H_
#define CALCULATECONTINGENCYTABLE_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>

namespace CuEira {
namespace CUDA {
namespace Kernel {

/**
 * This kernel calculates the table cells for the contingency table
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */__global__ void CalculateContingencyTable(const float* snpData, const float* envData, const float* phenotypeData,
    float* contigencyTable, const int length, const int numberOfBlocks){
  __shared__ float cache[256][8]; //Row major //TODO numberOfThreadsPerBlock make a config file with numbers etc?
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  cache[cacheIndex][0] = 0;
  cache[cacheIndex][1] = 0;
  cache[cacheIndex][2] = 0;
  cache[cacheIndex][3] = 0;
  cache[cacheIndex][4] = 0;
  cache[cacheIndex][5] = 0;
  cache[cacheIndex][6] = 0;
  cache[cacheIndex][7] = 0;

#pragma unroll 10
  for(int i = threadId * stepSize; i < threadId * stepSize + stepSize && i < length; ++i){
    ++cache[cacheIndex][(int) (phenotypeData[i] * 4 + snpData[i] + 2 * envData[i])];
    //NOTE be careful with unrolling since it can not go outside the length of actual data even if vectors are padded
  }

  __syncthreads();

  int i = blockDim.x / 2;
  while(i != 0){
    if(cacheIndex < i){
      cache[cacheIndex][0] += cache[cacheIndex + i][0];
      cache[cacheIndex][1] += cache[cacheIndex + i][1];
      cache[cacheIndex][2] += cache[cacheIndex + i][2];
      cache[cacheIndex][3] += cache[cacheIndex + i][3];

    }else if(cacheIndex >= i && cacheIndex < i * 2){
      cache[cacheIndex - i][4] += cache[cacheIndex][4];
      cache[cacheIndex - i][5] += cache[cacheIndex][5];
      cache[cacheIndex - i][6] += cache[cacheIndex][6];
      cache[cacheIndex - i][7] += cache[cacheIndex][7];
    }

    __syncthreads();
    i /= 2;
  }

  if(cacheIndex < 8){
    contigencyTable[blockIdx.x + numberOfBlocks * cacheIndex] = cache[0][cacheIndex];
  }
}

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* CALCULATECONTINGENCYTABLE_H_ */
