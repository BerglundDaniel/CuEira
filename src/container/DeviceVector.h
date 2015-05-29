#ifndef DEVICEVECTOR_H_
#define DEVICEVECTOR_H_

#include <math.h>

#include <DimensionMismatch.h>
#include <CudaAdapter.cu>

namespace CuEira {
namespace Container {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DeviceVector {
public:
  __host__ explicit DeviceVector(int numberOfRows);
  DeviceVector(int numberOfRealRows, int numberOfRows, PRECISION* vectorDevice);
  virtual ~DeviceVector();

  __device__ __host__ int getNumberOfRows() const;
  __device__ __host__ int getNumberOfColumns() const;

  __device__ __host__ PRECISION* operator()(int row);
  __device__ __host__ const PRECISION* operator()(int row)const;

  __device__ __host__ int getRealNumberOfRows() const;
  __device__ __host__ int getRealNumberOfColumns() const;
  __host__ void updateSize(int numberOfRows);

  __device__ __host__ PRECISION* getMemoryPointer();
  __device__ __host__ const PRECISION* getMemoryPointer() const;

#ifndef __CUDACC__
  DeviceVector(const DeviceVector&) = delete;
  DeviceVector(DeviceVector&&) = delete;
  DeviceVector& operator=(const DeviceVector&) = delete;
  DeviceVector& operator=(DeviceVector&&) = delete;
#endif

private:
  const int numberOfRealRows;
  int numberOfRows;
  bool subview;
  PRECISION* vectorDevice;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* DEVICEVECTOR_H_ */
