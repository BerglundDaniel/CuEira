#ifndef DEVICEMATRIX_H_
#define DEVICEMATRIX_H_

#include <math.h>

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
  __host__ explicit DeviceMatrix(int numberOfRows, int numberOfColumns);
  virtual ~DeviceMatrix();

  __device__ __host__ int getNumberOfRows() const;
  __device__ __host__ int getNumberOfColumns() const;

  __host__ DeviceVector* operator()(int column);
  __host__ const DeviceVector* operator()(int column) const;

  __device__ __host__ PRECISION* operator()(int row, int column);
  __device__ __host__ const PRECISION* operator()(int row, int column) const;

  __device__ __host__ int getRealNumberOfRows() const;
  __device__ __host__ int getRealNumberOfColumns() const;
  __host__ void updateSize(int numberOfRows, int numberOfColumns);
  __host__ void updateNumberOfRows(int numberOfRows);
  __host__ void updateNumberOfColumns(int numberOfColumns);

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
  const int numberOfRealRows;
  const int numberOfRealColumns;
  int numberOfRows;
  int numberOfColumns;
  PRECISION* matrixDevice;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* DEVICEMATRIX_H_ */
