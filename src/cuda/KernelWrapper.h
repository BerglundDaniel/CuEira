#ifndef KERNELWRAPPER_H_
#define KERNELWRAPPER_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

#include <CudaAdapter.cu>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <CudaException.h>

namespace CuEira {
namespace CUDA {

using namespace CuEira::Container;

/**
 * This is a class that wraps the kernels in the kernels namespace. Each function assumes that symbols for the size have been set on the device.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class KernelWrapper {
public:
  /**
   * Constructor for the class. Takes the stream the transfers should be executed on. Some functions requires that a cublas context has been created.
   * All of them assumes a cuda context exists for the stream.
   */
  KernelWrapper(const cudaStream_t& cudaStream);
  virtual ~KernelWrapper();

  /**
   * Performs the logistic transform, exp(x)/(1+exp(x)), on each element in logitVector and stores the results in probabilities. Assumes both have a length of numberOfRows.
   */
  void logisticTransform(const DeviceVector& logitVector, DeviceVector& probabilites) const;

  /**
   * Divideds each element in numeratorVector with its corresponding element in denomitorVector. Assumes both have length numberOfPredictors.
   */
  void elementWiseDivision(const DeviceVector& numeratorVector, const DeviceVector& denomitorVector,
      DeviceVector& result) const;

  /**
   * Calculates all the parts of a loglikelihood. The sum of the elements in result is the loglikelihood. Assumes both have a length of numberOfRows.
   */
  void logLikelihoodParts(const DeviceVector& outcomesVector, const DeviceVector& probabilites,
      DeviceVector& result) const;

  /**
   * Calculates the absolute difference for each element. Assumes both have length numberOfPredictors.
   */
  void absoluteDifference(const DeviceVector& vector1, const DeviceVector& vector2, DeviceVector& result) const;

private:
  const cudaStream_t& cudaStream;
  static const int numberOfThreadsPerBlock = 256;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* KERNELWRAPPER_H_ */
