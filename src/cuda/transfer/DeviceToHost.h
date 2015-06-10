#ifndef DEVICETOHOST_H_
#define DEVICETOHOST_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <PinnedHostVector.h>
#include <PinnedHostMatrix.h>
#include <Stream.h>

namespace CuEira {
namespace CUDA {

using namespace CuEira::Container;

/**
 * This wraps the CUDA functionallity of transfers from the device to the host
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

/**
 * Creates a matrix on the host and copies the data from the device matrix without releasing it, the call is made on the stream that was provided in the constructor.
 */
PinnedHostMatrix* transferMatrix(const Stream& stream, const DeviceMatrix& matrixDevice);

/**
 * Creates a vector on the host and copies the data from the device vector without releasing it, the call is made on the stream that was provided in the constructor.
 */
PinnedHostVector* transferVector(const Stream& stream, const DeviceVector& vectorDevice);

/**
 * Copies the data from the device matrix to the specified point in host memory without releasing the device memory, the call is made on the stream that
 * was provided in the constructor. The host memory has to be previously allocated as pinned host memory.
 */
void transferMatrix(const Stream& stream, const DeviceMatrix& matrixDevice, PRECISION* hostMemoryPosition);

/**
 * Copies the data from the device vector to the specified point in host memory without releasing the device memory, the call is made on the stream that
 * was provided in the constructor. The host memory has to be previously allocated as pinned host memory.
 */
void transferVector(const Stream& stream, const DeviceVector& vectorDevice, PRECISION* hostMemoryPosition);

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* DEVICETOHOST_H_ */
