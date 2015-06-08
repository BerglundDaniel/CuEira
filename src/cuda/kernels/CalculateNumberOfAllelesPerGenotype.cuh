#ifndef CALCULATENUMBEROFALLELESPERGENOTYPE_H_
#define CALCULATENUMBEROFALLELESPERGENOTYPE_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>

namespace CuEira {
namespace CUDA {
namespace Kernel {

const int stepSize = 10;

/**
 * This kernel calculates the number of alleles per genotype and phenotype group, which is 6 different combinations
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */__global__ void CalculateNumberOfAllelesPerGenotype(const float* snpData, const float* phenotypeData,
    float* numberOfAllelesPerGenotype, const int length, const int lengthAllelesPerGenotype){
  __shared__ float cache[256][6]; //Row major //TODO numberOfThreadsPerBlock make a config file with numbers etc?
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  cache[cacheIndex][0] = 0;
  cache[cacheIndex][1] = 0;
  cache[cacheIndex][2] = 0;
  cache[cacheIndex][3] = 0;
  cache[cacheIndex][4] = 0;
  cache[cacheIndex][5] = 0;

  //UNROLL GPU
  for(int i = threadId * stepSize; i < threadId * stepSize + stepSize && i < length; ++i){
    ++cache[cacheIndex][(int) (snpData[i] + 3 * phenotypeData[i])]; //FIXME
  }

  __syncthreads();

  int i = blockDim.x / 2;
  while(i != 0){
    if(cacheIndex < i){
      cache[cacheIndex][0] += cache[cacheIndex + i][0];
      cache[cacheIndex][1] += cache[cacheIndex + i][1];
      cache[cacheIndex][2] += cache[cacheIndex + i][2];

    }else if(cacheIndex >= i && cacheIndex < i * 2){
      cache[cacheIndex - i][3] += cache[cacheIndex][3];
      cache[cacheIndex - i][4] += cache[cacheIndex][4];
      cache[cacheIndex - i][5] += cache[cacheIndex][5];
    }

    __syncthreads();
    i /= 2;
  }

  if(cacheIndex < 6){
    numberOfAllelesPerGenotype[blockIdx.x + lengthAllelesPerGenotype * cacheIndex] = cache[0][cacheIndex]; //FIXME
  }
}

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* CALCULATENUMBEROFALLELESPERGENOTYPE_H_ */
