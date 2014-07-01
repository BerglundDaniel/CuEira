#ifndef KERNELWRAPPER_H_
#define KERNELWRAPPER_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <sstream>
#include <string>

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
  KernelWrapper(const cudaStream_t& cudaStream, const cublasHandle_t& cublasHandle);
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
   * Adds each element result(i)=vector1(i) + vector2(i)
   */
  void elementWiseAddition(const DeviceVector& vector1, const DeviceVector& vector2, DeviceVector& result) const;

  /**
   * Multiplies each element result(i)=vector1(i) * vector2(i)
   */
  void elementWiseMultiplication(const DeviceVector& vector1, const DeviceVector& vector2, DeviceVector& result) const;

  /**
   * Calculates all the parts of a loglikelihood. The sum of the elements in result is the loglikelihood. Assumes both have a length of numberOfRows.
   */
  void logLikelihoodParts(const DeviceVector& outcomesVector, const DeviceVector& probabilites,
      DeviceVector& result) const;

  /**
   * Calculates the absolute difference for each element. Assumes both have length numberOfPredictors.
   */
  void elementWiseAbsoluteDifference(const DeviceVector& vector1, const DeviceVector& vector2,
      DeviceVector& result) const;

  /**
   * Copies from vectorFrom to vectorTo element wise
   */
  void copyVector(const DeviceVector& vectorFrom, DeviceVector& vectorTo) const;

  /**
   * Calculates x*(1-x) for each element
   */
  void probabilitesMultiplyProbabilites(const DeviceVector& probabilitesDevice, DeviceVector& result) const;

  /**
   * Calculates vector1-vector2
   */
  void elementWiseDifference(const DeviceVector& vector1, const DeviceVector& vector2, DeviceVector& result) const;

  /**
   * Asdf
   */
  void matrixVectorMultiply(const DeviceMatrix& matrix, const DeviceVector& vector, DeviceVector& result) const;

  /**
   * Asdf
   */
  void matrixTransVectorMultiply(const DeviceMatrix& matrix, const DeviceVector& vector, DeviceVector& result) const;

  /**
   * Asdf
   */
  void matrixTransMatrixMultiply(const DeviceMatrix& matrix1, const DeviceMatrix& matrix2, DeviceMatrix& result) const;

  /**
   * Asdf
   */
  void columnByColumnMatrixVectorElementWiseMultiply(const DeviceMatrix& matrix, const DeviceVector& vector,
      DeviceMatrix& result) const;

  /**
   * Sums the vectors elements and puts the result in the given pointer
   */
  void sumResultToHost(const DeviceVector& vector, const DeviceVector& oneVector, PRECISION* sumHost) const;

  /**
   * Syncs the associated stream
   */
  inline void syncStream() const {
    cudaStreamSynchronize(cudaStream);
  }

  /**
   * Set the device symbol
   */
  void setSymbolNumberOfRows(int numberOfRows) const;

  /**
   * Set the device symbol
   */
  void setSymbolNumberOfPredictors(int numberOfPredictors) const;

private:
  const cublasHandle_t& cublasHandle;
  const cudaStream_t& cudaStream;
  static const int numberOfThreadsPerBlock = 256;

  const PRECISION* constOne;
  const PRECISION* constZero;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* KERNELWRAPPER_H_ */
