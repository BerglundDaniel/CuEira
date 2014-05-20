#ifndef DEVICEVECTOR_H_
#define DEVICEVECTOR_H_

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <CudaAdapter.cu>

namespace CuEira {
namespace CUDA {
class DeviceToHost;
class HostToDevice;
}
namespace Container {
class DeviceMatrix;
class DeviceVectorTest;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DeviceVector {
  friend DeviceMatrix;
  friend DeviceVectorTest;
  friend CUDA::DeviceToHost;
  friend CUDA::HostToDevice;
  FRIEND_TEST(DeviceVectorTest, AccessOperator);
public:
  DeviceVector(int numberOfRows);
  virtual ~DeviceVector();

  __device__ __host__ int getNumberOfRows() const;
  __device__ __host__ int getNumberOfColumns() const;

  __device__ __host__ PRECISION* operator()(unsigned int row);
  __device__ __host__ const PRECISION* operator()(unsigned int row)const;

protected:
  DeviceVector(int numberOfRows, PRECISION* vectorDevice);
  __device__ __host__ PRECISION* getMemoryPointer();
  __device__ __host__ const PRECISION* getMemoryPointer() const;

private:
  const int numberOfRows;
  const int numberOfColumns;
  bool subview;
  PRECISION* vectorDevice;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* DEVICEVECTOR_H_ */
