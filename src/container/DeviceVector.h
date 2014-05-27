#ifndef DEVICEVECTOR_H_
#define DEVICEVECTOR_H_

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
  DeviceVector(int numberOfRows);
  DeviceVector(int numberOfRows, PRECISION* vectorDevice);
  virtual ~DeviceVector();

  __device__ __host__ int getNumberOfRows() const;
  __device__ __host__ int getNumberOfColumns() const;

  __device__ __host__ PRECISION* operator()(unsigned int row);
  __device__ __host__ const PRECISION* operator()(unsigned int row)const;

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