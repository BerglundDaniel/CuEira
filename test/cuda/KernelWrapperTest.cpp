#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>

#include <CudaAdapter.cu>
#include <CudaException.h>
#include <CublasException.h>
#include <KernelWrapper.h>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <PinnedHostMatrix.h>
#include <PinnedHostVector.h>
#include <DeviceToHost.h>
#include <HostToDevice.h>
#include <Device.h>
#include <Stream.h>
#include <StreamFactory.h>

namespace CuEira {
namespace CUDA {

/**
 * Test for KernelWrapper
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class KernelWrapperTest: public ::testing::Test {
protected:
  KernelWrapperTest();
  virtual ~KernelWrapperTest();
  virtual void SetUp();
  virtual void TearDown();

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
  KernelWrapper kernelWrapper;
};

KernelWrapperTest::KernelWrapperTest() :
    device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDeviceStream1(*stream), deviceToHostStream1(
        *stream), kernelWrapper(*stream) {

}

KernelWrapperTest::~KernelWrapperTest() {
  delete stream;
}

void KernelWrapperTest::SetUp() {

}

void KernelWrapperTest::TearDown() {

}

TEST_F(KernelWrapperTest, ColumnByColumnMatrixVectorElementWiseMultiply) {
  const int numberOfRows = 3;
  const int numberOfColumns = 2;

  PinnedHostMatrix matrix(numberOfRows, numberOfColumns);
  PinnedHostVector vector(numberOfRows);

  matrix(0, 0) = 0;
  matrix(1, 0) = 1;
  matrix(2, 0) = 2;

  matrix(0, 1) = 3;
  matrix(1, 1) = 4;
  matrix(2, 1) = 5;

  vector(0) = 1;
  vector(1) = 2;
  vector(2) = 3;

  DeviceMatrix* matrixDevice = hostToDeviceStream1.transferMatrix(matrix);
  DeviceVector* vectorDevice = hostToDeviceStream1.transferVector(vector);
  DeviceMatrix* resultDevice = new DeviceMatrix(numberOfRows, numberOfColumns);

  kernelWrapper.columnByColumnMatrixVectorElementWiseMultiply(*matrixDevice, *vectorDevice, *resultDevice);
  kernelWrapper.syncStream();
  handleCudaStatus(cudaGetLastError(),
      "Error with ColumnByColumnMatrixVectorElementWiseMultiply in ColumnByColumnMatrixVectorElementWiseMultiplyTest: ");

  HostMatrix* resultHost = deviceToHostStream1.transferMatrix(*resultDevice);
  kernelWrapper.syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with transfer in ColumnByColumnMatrixVectorElementWiseMultiply: ");

  EXPECT_EQ(0, (*resultHost)(0, 0));
  EXPECT_EQ(2, (*resultHost)(1, 0));
  EXPECT_EQ(6, (*resultHost)(2, 0));

  EXPECT_EQ(3, (*resultHost)(0, 1));
  EXPECT_EQ(8, (*resultHost)(1, 1));
  EXPECT_EQ(15, (*resultHost)(2, 1));

  delete matrixDevice;
  delete vectorDevice;
  delete resultDevice;
  delete resultHost;
}

}
/* namespace CUDA */
} /* namespace CuEira */

