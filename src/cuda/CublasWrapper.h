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
namespace Kernel {

/**
 * This wraps the CUBLAS library
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

/**
 * Copies from vectorFrom to vectorTo element wise
 */
void copyVector(const Stream& stream, const DeviceVector& vectorFrom, DeviceVector& vectorTo);

/**
 * Asdf
 */
void matrixVectorMultiply(const Stream& stream, const DeviceMatrix& matrix, const DeviceVector& vector,
    DeviceVector& result);

/**
 * Asdf
 */
void matrixTransVectorMultiply(const Stream& stream, const DeviceMatrix& matrix, const DeviceVector& vector,
    DeviceVector& result);

/**
 * Asdf
 */
void matrixTransMatrixMultiply(const Stream& stream, const DeviceMatrix& matrix1, const DeviceMatrix& matrix2,
    DeviceMatrix& result);

/**
 * Sums the vectors elements and puts the result in the given pointer
 */
void sumResultToHost(const Stream& stream, const DeviceVector& vector, const DeviceVector& oneVector,
    PRECISION& sumHost);

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* CUBLASWRAPPER_H_ */
