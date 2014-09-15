#ifndef DEVICEMATRIX_H_
#define DEVICEMATRIX_H_

#include <DimensionMismatch.h>
#include <CudaAdapter.cu>
#include <DeviceVector.h>

namespace CuEira {
namespace Container {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DeviceMatrix {

public:
  DeviceMatrix(int numberOfRows, int numberOfColumns);
  DeviceMatrix(int numberOfRows, int numberOfColumns, PRECISION* matrixDevice);
  virtual ~DeviceMatrix();

  __device__ __host__ int getNumberOfRows() const;
  __device__ __host__ int getNumberOfColumns() const;

  DeviceVector* operator()(unsigned int column);
  const DeviceVector* operator()(unsigned int column) const;

  __device__ __host__ PRECISION* operator()(unsigned int row, unsigned int column);
  __device__ __host__ const PRECISION* operator()(unsigned int row, unsigned int column) const;

  __device__ __host__ PRECISION* getMemoryPointer();
  __device__ __host__ const PRECISION* getMemoryPointer() const;

#ifndef __CUDACC__
  DeviceMatrix(const DeviceMatrix&) = delete;
  DeviceMatrix(DeviceMatrix&&) = delete;
  DeviceMatrix& operator=(const DeviceMatrix&) = delete;
  DeviceMatrix& operator=(DeviceMatrix&&) = delete;
#endif

private:
  const bool subview;
  const int numberOfRows;
  const int numberOfColumns;
  PRECISION* matrixDevice;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* DEVICEMATRIX_H_ */
