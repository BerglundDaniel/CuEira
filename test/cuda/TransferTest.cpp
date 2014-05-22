#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>
#include <CudaException.h>
#include <CublasException.h>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <PinnedHostMatrix.h>
#include <PinnedHostVector.h>
#include <DeviceToHost.h>
#include <HostToDevice.h>

namespace CuEira {
namespace CUDA {

/**
 * Test for testing transfers between host and device in both directions.
 * Assumes that the container classes are working.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class TransferTest: public ::testing::Test {
protected:
  TransferTest();
  virtual ~TransferTest();
  virtual void SetUp();
  virtual void TearDown();

  cublasStatus_t cublasStatus;
  cudaStream_t stream1;
  cublasHandle_t cublasHandle;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
};

TransferTest::TransferTest() :
    cublasStatus(cublasCreate(&cublasHandle)), hostToDeviceStream1(HostToDevice(stream1)), deviceToHostStream1(
        DeviceToHost(stream1)) {

  handleCublasStatus(cublasStatus, "Failed to create cublas handle:");
  handleCudaStatus(cudaStreamCreate(&stream1), "Failed to create cuda stream 1:");
  handleCublasStatus(cublasSetStream(cublasHandle, stream1), "Failed to set cuda stream:");
}

TransferTest::~TransferTest() {
  handleCublasStatus(cublasDestroy(cublasHandle), "Failed to destroy cublas handle:");
  handleCudaStatus(cudaStreamDestroy(stream1), "Failed to destroy cuda stream 1:");
}

void TransferTest::SetUp() {

}

void TransferTest::TearDown() {

}

TEST_F(TransferTest, TransferVector) {
  const int numberOfRows = 5;

  Container::PinnedHostVector* hostVectorFrom = new Container::PinnedHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    (*hostVectorFrom)(i) = i + 10;
  }

  Container::DeviceVector* deviceVector = hostToDeviceStream1.transferVector(hostVectorFrom);
  Container::HostVector* hostVectorTo = deviceToHostStream1.transferVector(deviceVector);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error when transferring vector in test: ");

  ASSERT_EQ(numberOfRows, hostVectorTo->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i + 10, (*hostVectorFrom)(i));
    EXPECT_EQ(i + 10, (*hostVectorTo)(i));
  }

  delete hostVectorFrom;
  delete deviceVector;
  delete hostVectorTo;
}

TEST_F(TransferTest, TransferMatrix) {
  const int numberOfRows = 5;
  const int numberOfColumns = 3;

  Container::PinnedHostMatrix* hostMatrixFrom = new Container::PinnedHostMatrix(numberOfRows, numberOfColumns);
  for(int j = 0; j < numberOfColumns; ++j){
    for(int i = 0; i < numberOfRows; ++i){
      (*hostMatrixFrom)(i, j) = i + (10 * j);
    }
  }

  Container::DeviceMatrix* deviceMatrix = hostToDeviceStream1.transferMatrix(hostMatrixFrom);
  Container::HostMatrix* hostMatrixTo = deviceToHostStream1.transferMatrix(deviceMatrix);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error when transferring matrix in test: ");

  ASSERT_EQ(numberOfRows, hostMatrixTo->getNumberOfRows());
  ASSERT_EQ(numberOfColumns, hostMatrixTo->getNumberOfColumns());

  for(int j = 0; j < numberOfColumns; ++j){
    for(int i = 0; i < numberOfRows; ++i){
      EXPECT_EQ(i + (10 * j), (*hostMatrixFrom)(i, j));
      EXPECT_EQ(i + (10 * j), (*hostMatrixTo)(i, j));
    }
  }

  delete deviceMatrix;
  delete hostMatrixTo;
}

TEST_F(TransferTest, TransferVectorCustomPoint) {
  const int numberOfRows = 5;
  const int numberOfRowsBigMatrix = 10;
  const int numberOfColumnsBigMatrix = 10;

  Container::PinnedHostMatrix* hostMatrixBig = new Container::PinnedHostMatrix(numberOfRowsBigMatrix,
      numberOfColumnsBigMatrix);
  for(int j = 0; j < numberOfColumnsBigMatrix; ++j){
    for(int i = 0; i < numberOfRowsBigMatrix; ++i){
      (*hostMatrixBig)(i, j) = 1;
    }
  }

  Container::PinnedHostVector* hostVectorFrom = new Container::PinnedHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    (*hostVectorFrom)(i) = i + 10;
  }

  Container::DeviceMatrix* deviceMatrixBig = hostToDeviceStream1.transferMatrix(hostMatrixBig);
  PRECISION* vectorPos = deviceMatrixBig->getMemoryPointer();
  hostToDeviceStream1.transferVector(hostVectorFrom, vectorPos);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error when transferring to device in TransferTest: ");

  Container::DeviceVector* deviceVector = new Container::DeviceVector(numberOfRows, vectorPos);
  Container::HostVector* hostVectorTo = deviceToHostStream1.transferVector(deviceVector);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error when transferring from device in TransferTest: ");

  ASSERT_EQ(numberOfRows, hostVectorTo->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i + 10, (*hostVectorTo)(i));
  }

  delete hostVectorFrom;
  delete hostMatrixBig;
  delete hostVectorTo;
  delete deviceVector;
}

TEST_F(TransferTest, TransferMatrixCustomPoint) {
//TODO
}

}
/* namespace Container */
} /* namespace CUDA */

