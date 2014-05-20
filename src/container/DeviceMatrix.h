#ifndef DEVICEMATRIX_H_
#define DEVICEMATRIX_H_

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <CudaAdapter.cu>
#include <DeviceVector.h>

namespace CuEira {
namespace CUDA {
class DeviceToHost;
class HostToDevice;
}
namespace Container {
class DeviceMatrixTest;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DeviceMatrix {
  friend DeviceMatrixTest;
  friend CUDA::DeviceToHost;
  friend CUDA::HostToDevice;
  FRIEND_TEST(DeviceMatrixTest, AccessOperator);
public:
  DeviceMatrix(int numberOfRows, int numberOfColumns);
  virtual ~DeviceMatrix();

  __device__ __host__ int getNumberOfRows() const;
  __device__ __host__ int getNumberOfColumns() const;

  DeviceVector* operator()(unsigned int column);
  const DeviceVector* operator()(unsigned int column) const;

  __device__ __host__ PRECISION* operator()(unsigned int row, unsigned int column);
  __device__ __host__ const PRECISION* operator()(unsigned int row, unsigned int column) const;

protected:
  DeviceMatrix(int numberOfRows, int numberOfColumns, PRECISION* matrixDevice);
  __device__ __host__ PRECISION* getMemoryPointer();
  __device__ __host__ const PRECISION* getMemoryPointer() const;

private:
  const bool subview;
  const int numberOfRows;
  const int numberOfColumns;
  PRECISION* matrixDevice;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* DEVICEMATRIX_H_ */
