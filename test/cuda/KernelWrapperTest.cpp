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

TEST_F(KernelWrapperTest, matrixTransMatrixMultiply) {
  const int numberOfRows = 5;
  const int numberOfColumns = 3;
  const int numberOfRows2 = 5;
  const int numberOfColumns2 = 4;
  PinnedHostMatrix matrixT(numberOfRows, numberOfColumns);
  PinnedHostMatrix matrix2(numberOfRows2, numberOfColumns2);

  matrixT(0, 0) = 1;
  matrixT(1, 0) = 2;
  matrixT(2, 0) = 3;
  matrixT(3, 0) = 4;
  matrixT(4, 0) = 5;

  matrixT(0, 1) = 10;
  matrixT(1, 1) = 20;
  matrixT(2, 1) = 30;
  matrixT(3, 1) = 40;
  matrixT(4, 1) = 50;

  matrixT(0, 2) = 1.1;
  matrixT(1, 2) = 2.2;
  matrixT(2, 2) = 3.3;
  matrixT(3, 2) = 4.4;
  matrixT(4, 2) = 5.5;

  for(int i = 0; i < numberOfRows2; ++i){
    matrix2(i, 0) = 6;
  }

  for(int i = 0; i < numberOfRows2; ++i){
    matrix2(i, 1) = 7;
  }

  for(int i = 0; i < numberOfRows2; ++i){
    matrix2(i, 2) = 8;
  }

  for(int i = 0; i < numberOfRows2; ++i){
    matrix2(i, 3) = 9;
  }

  DeviceMatrix* matrixTDevice = hostToDeviceStream1.transferMatrix(matrixT);
  DeviceMatrix* matrix2Device = hostToDeviceStream1.transferMatrix(matrix2);
  DeviceMatrix* resultDevice = new DeviceMatrix(numberOfColumns, numberOfColumns2);

  kernelWrapper.matrixTransMatrixMultiply(*matrixTDevice, *matrix2Device, *resultDevice);
  kernelWrapper.syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with matrixTransMatrixMultiply in matrixTransMatrixMultiply: ");

  HostMatrix* resultHost = deviceToHostStream1.transferMatrix(*resultDevice);
  kernelWrapper.syncStream();
  handleCudaStatus(cudaGetLastError(), "Error with transfer in matrixTransMatrixMultiply: ");

  EXPECT_EQ(90, (*resultHost)(0, 0));
  EXPECT_EQ(105, (*resultHost)(0, 1));
  EXPECT_EQ(120, (*resultHost)(0, 2));
  EXPECT_EQ(135, (*resultHost)(0, 3));

  EXPECT_EQ(900, (*resultHost)(1, 0));
  EXPECT_EQ(1050, (*resultHost)(1, 1));
  EXPECT_EQ(1200, (*resultHost)(1, 2));
  EXPECT_EQ(1350, (*resultHost)(1, 3));

  EXPECT_EQ(99, (*resultHost)(2, 0));
  EXPECT_EQ(115.5, (*resultHost)(2, 1));
  EXPECT_EQ(132, (*resultHost)(2, 2));
  EXPECT_EQ(148.5, (*resultHost)(2, 3));

  delete matrixTDevice;
  delete matrix2Device;
  delete resultDevice;
  delete resultHost;
}

}
/* namespace CUDA */
} /* namespace CuEira */

