#ifndef HOSTTODEVICE_H_
#define HOSTTODEVICE_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <PinnedHostVector.h>
#include <PinnedHostMatrix.h>
#include <Stream.h>

namespace CuEira {
namespace CUDA {

using namespace CuEira::Container;

/**
 * This wraps the CUDA functionallity of transfers from the host to the device
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

/**
 * Creates a matrix on the devices and copies the data from the matrix on the host, the call is made on the stream that was provided in the constructor.
 * Does not release the memory on the device.
 */
DeviceMatrix* transferMatrix(const Stream& stream, const PinnedHostMatrix& matrixHost);

/**
 * Creates a vector on the devices and copies the data from the vector on the host, the call is made on the stream that was provided in the constructor.
 * Does not release the memory on the device.
 */
DeviceVector* transferVector(const Stream& stream, const PinnedHostVector& vectorHost);

/**
 * Transfers the matrix to the area in device memory given by deviceMemoryPosition. It has to point to an area with already allocated memory and with
 * enough space to fit the matrix after the point specified.
 */
void transferMatrix(const Stream& stream, const PinnedHostMatrix& matrixHost, PRECISION* deviceMemoryPosition);

/**
 * Transfers the vector to the area in device memory given by deviceMemoryPosition. It has to point to an area with already allocated memory and with
 * enough space to fit the vector after the point specified.
 */
void transferVector(const Stream& stream, const PinnedHostVector& vectorHost, PRECISION* deviceMemoryPosition);

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* HOSTTODEVICE_H_ */
