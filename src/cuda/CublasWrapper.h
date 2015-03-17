#ifndef CUBLASWRAPPER_H_
#define CUBLASWRAPPER_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <sstream>
#include <string>

#include <CudaAdapter.cu>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <CudaException.h>
#include <CublasException.h>
#include <Stream.h>

namespace CuEira {
namespace CUDA {

/**
 * This wraps the CUBLAS library
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CublasWrapper {
public:
  /**
   * Constructor for the class.
   */
  explicit CublasWrapper(const Stream& stream);
  virtual ~CublasWrapper();

  /**
   * Copies from vectorFrom to vectorTo element wise
   */
  void copyVector(const DeviceVector& vectorFrom, DeviceVector& vectorTo) const;

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
   * Sums the vectors elements and puts the result in the given pointer
   */
  void sumResultToHost(const DeviceVector& vector, const DeviceVector& oneVector, PRECISION& sumHost) const;

  /**
   * Syncs the associated stream
   */
  inline void syncStream() const {
    stream.syncStream();
  }

private:
  const Stream& stream;
  const cublasHandle_t& cublasHandle;
  const cudaStream_t& cudaStream;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* CUBLASWRAPPER_H_ */
