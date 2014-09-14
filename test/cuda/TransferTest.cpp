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
#include <Device.h>
#include <Stream.h>
#include <StreamFactory.h>

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

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
};

TransferTest::TransferTest() :
    device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDeviceStream1(*stream), deviceToHostStream1(
        *stream) {

}

TransferTest::~TransferTest() {
  delete stream;
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
  cudaStreamSynchronize (stream1);
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
  cudaStreamSynchronize (stream1);
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

TEST_F(TransferTest, TransferVectorCustomPointDevice) {
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
  PRECISION* vectorPos = deviceMatrixBig->getMemoryPointer() + 3;
  hostToDeviceStream1.transferVector(hostVectorFrom, vectorPos);
  cudaStreamSynchronize (stream1);
  handleCudaStatus(cudaGetLastError(), "Error when transferring to device in TransferTest: ");

  Container::DeviceVector* deviceVector = new Container::DeviceVector(numberOfRows, vectorPos);
  Container::HostVector* hostVectorTo = deviceToHostStream1.transferVector(deviceVector);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error when transferring from device in TransferTest: ");

  ASSERT_EQ(numberOfRows, hostVectorTo->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(i + 10, (*hostVectorTo)(i));
  }

  delete deviceMatrixBig;
  delete hostVectorFrom;
  delete hostMatrixBig;
  delete hostVectorTo;
  delete deviceVector;
}

TEST_F(TransferTest, TransferMatrixCustomPointDevice) {
  const int numberOfRows = 5;
  const int numberOfColumns = 4;
  const int numberOfRowsBigMatrix = 10;
  const int numberOfColumnsBigMatrix = 10;

  Container::PinnedHostMatrix* hostMatrixBig = new Container::PinnedHostMatrix(numberOfRowsBigMatrix,
      numberOfColumnsBigMatrix);
  for(int j = 0; j < numberOfColumnsBigMatrix; ++j){
    for(int i = 0; i < numberOfRowsBigMatrix; ++i){
      (*hostMatrixBig)(i, j) = 1;
    }
  }

  Container::PinnedHostMatrix* hostMatrixFrom = new Container::PinnedHostMatrix(numberOfRows, numberOfColumns);
  for(int j = 0; j < numberOfColumns; ++j){
    for(int i = 0; i < numberOfRows; ++i){
      (*hostMatrixFrom)(i, j) = i + (j * numberOfRows);
    }
  }

  Container::DeviceMatrix* deviceMatrixBig = hostToDeviceStream1.transferMatrix(hostMatrixBig);
  PRECISION* matrixPos = deviceMatrixBig->getMemoryPointer() + 3;
  hostToDeviceStream1.transferMatrix(hostMatrixFrom, matrixPos);
  cudaStreamSynchronize (stream1);
  handleCudaStatus(cudaGetLastError(), "Error when transferring to device in TransferTest: ");

  Container::DeviceMatrix* deviceMatrix = new Container::DeviceMatrix(numberOfRows, numberOfColumns, matrixPos);
  Container::HostMatrix* hostMatrixTo = deviceToHostStream1.transferMatrix(deviceMatrix);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error when transferring from device in TransferTest: ");

  ASSERT_EQ(numberOfRows, hostMatrixTo->getNumberOfRows());
  ASSERT_EQ(numberOfColumns, hostMatrixTo->getNumberOfColumns());

  for(int j = 0; j < numberOfColumns; ++j){
    for(int i = 0; i < numberOfRows; ++i){
      EXPECT_EQ(i + (j * numberOfRows), (*hostMatrixTo)(i, j));
    }
  }

  delete deviceMatrixBig;
  delete hostMatrixFrom;
  delete hostMatrixBig;
  delete hostMatrixTo;
  delete deviceMatrix;
}

TEST_F(TransferTest, TransferVectorCustomPointHost) {
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

  Container::DeviceVector* deviceVector = hostToDeviceStream1.transferVector(hostVectorFrom);
  cudaStreamSynchronize (stream1);
  handleCudaStatus(cudaGetLastError(), "Error when transferring to device in TransferTest: ");

  int offset = 2;
  PRECISION* vectorPos = hostMatrixBig->getMemoryPointer() + offset;
  deviceToHostStream1.transferVector(deviceVector, vectorPos);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error when transferring from device in TransferTest: ");

  for(int j = 0; j < numberOfColumnsBigMatrix; ++j){
    for(int i = 0; i < numberOfRowsBigMatrix; ++i){
      if(i >= offset && i < (offset + numberOfRows) && j == 0){
        EXPECT_EQ((*hostVectorFrom)(i - offset), (*hostMatrixBig)(i, j));
      }else{
        EXPECT_EQ(1, (*hostMatrixBig)(i, j));
      }
    }
  }

  delete hostVectorFrom;
  delete hostMatrixBig;
  delete deviceVector;
}

TEST_F(TransferTest, TransferMatrixCustomPointHost) {
  const int numberOfRows = 5;
  const int numberOfColumns = 4;
  const int numberOfRowsBigMatrix = 10;
  const int numberOfColumnsBigMatrix = 10;

  Container::PinnedHostMatrix* hostMatrixBig = new Container::PinnedHostMatrix(numberOfRowsBigMatrix,
      numberOfColumnsBigMatrix);
  for(int j = 0; j < numberOfColumnsBigMatrix; ++j){
    for(int i = 0; i < numberOfRowsBigMatrix; ++i){
      (*hostMatrixBig)(i, j) = 1;
    }
  }

  Container::PinnedHostMatrix* hostMatrixFrom = new Container::PinnedHostMatrix(numberOfRows, numberOfColumns);
  for(int j = 0; j < numberOfColumns; ++j){
    for(int i = 0; i < numberOfRows; ++i){
      (*hostMatrixFrom)(i, j) = i + (j * numberOfRows) + 1;
    }
  }

  Container::DeviceMatrix* deviceMatrix = hostToDeviceStream1.transferMatrix(hostMatrixFrom);
  cudaStreamSynchronize (stream1);
  handleCudaStatus(cudaGetLastError(), "Error when transferring to device in TransferTest: ");

  int offset = 3;
  PRECISION* matrixPos = hostMatrixBig->getMemoryPointer() + offset;

  deviceToHostStream1.transferMatrix(deviceMatrix, matrixPos);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error when transferring from device in TransferTest: ");

  int totalSize = numberOfRowsBigMatrix * numberOfColumnsBigMatrix;
  int matrixFromTotalSize = numberOfRows * numberOfColumns;
  PRECISION* matrixFromPointer = hostMatrixFrom->getMemoryPointer();
  PRECISION* matrixBigPointer = hostMatrixBig->getMemoryPointer();

  for(int ij = 0; ij < totalSize; ++ij){
    if(ij >= offset && ij < (offset + matrixFromTotalSize)){
      EXPECT_EQ(*(matrixFromPointer + ij - offset), *(matrixBigPointer + ij));
    }else{
      EXPECT_EQ(1, *(matrixBigPointer + ij));
    }
  }

  delete hostMatrixFrom;
  delete hostMatrixBig;
  delete deviceMatrix;
}

}
/* namespace CUDA */
} /* namespace CuEira*/

