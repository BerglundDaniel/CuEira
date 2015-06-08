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
#include <Stream.h>

namespace CuEira {
namespace CUDA {

using namespace CuEira::Container;

/**
 * This is a class that wraps the kernels in the kernels namespace.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class KernelWrapper {
public:
  /**
   * Constructor for the class.
   */
  explicit KernelWrapper(const Stream& stream);
  virtual ~KernelWrapper();

  /**
   * Performs the logistic transform, exp(x)/(1+exp(x)), on each element in logitVector and stores the results in probabilities
   */
  void logisticTransform(const DeviceVector& logitVector, DeviceVector& probabilites) const;

  /**
   * Divides each element in numeratorVector with its corresponding element in denomitorVector
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
   * Calculates all the parts of a loglikelihood. The sum of the elements in result is the loglikelihood
   */
  void logLikelihoodParts(const DeviceVector& outcomesVector, const DeviceVector& probabilites,
      DeviceVector& result) const;

  /**
   * Calculates the absolute difference for each element
   */
  void elementWiseAbsoluteDifference(const DeviceVector& vector1, const DeviceVector& vector2,
      DeviceVector& result) const;

  /**
   * Calculates x*(1-x) for each element
   */
  void probabilitesMultiplyProbabilites(const DeviceVector& probabilitesDevice, DeviceVector& result) const;

  /**
   * Calculates vector1-vector2
   */
  void elementWiseDifference(const DeviceVector& vector1, const DeviceVector& vector2, DeviceVector& result) const;

  //to=from[indexes[i]]
  void vectorCopyIndexes(const DeviceVector& indexes, const DeviceVector& from, DeviceVector& to) const;

  //c-vector[i]
  void constSubtractVector(const int c, DeviceVector& vector) const;

  //to[i] = snpToRisk[from[i]]
  void applyGeneticModel(const int snpToRisk[3], const DeviceVector& from, DeviceVector& to) const;

  //interaction=vector1*vector2 vector1=vector2=0 if interaction!=0
  void applyAdditiveModel(DeviceVector& vector1, DeviceVector& vector2, DeviceVector& interaction) const;

  //numberOfAllelesPerGenotype=snpData(i)+3*phenotypeData(i)
  Container::DeviceMatrix* calculateNumberOfAllelesPerGenotype(const Container::DeviceVector& snpData,
      const Container::DeviceVector& phenotypeData) const;

  /**
   * Syncs the associated stream
   */
  inline void syncStream() const{
    stream.syncStream();
  }

  const Stream& stream;

private:
  const cublasHandle_t& cublasHandle;
  const cudaStream_t& cudaStream;

  const PRECISION* constOne;
  const PRECISION* constZero;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* KERNELWRAPPER_H_ */
