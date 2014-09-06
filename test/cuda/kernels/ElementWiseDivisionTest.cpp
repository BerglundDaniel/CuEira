#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

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

namespace CuEira {
namespace CUDA {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ElementWiseDivisionTest: public ::testing::Test {
protected:
  ElementWiseDivisionTest();
  virtual ~ElementWiseDivisionTest();
  virtual void SetUp();
  virtual void TearDown();

  cublasStatus_t cublasStatus;
  cudaStream_t stream1;
  cublasHandle_t cublasHandle;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
  KernelWrapper kernelWrapper;
};

ElementWiseDivisionTest::ElementWiseDivisionTest() :
    cublasStatus(cublasCreate(&cublasHandle)), hostToDeviceStream1(HostToDevice(stream1)), deviceToHostStream1(
        DeviceToHost(stream1)), kernelWrapper(stream1, cublasHandle) {

  handleCublasStatus(cublasStatus, "Failed to create cublas handle:");
  handleCudaStatus(cudaStreamCreate(&stream1), "Failed to create cuda stream 1:");
  handleCublasStatus(cublasSetStream(cublasHandle, stream1), "Failed to set cuda stream:");
}

ElementWiseDivisionTest::~ElementWiseDivisionTest() {
  handleCublasStatus(cublasDestroy(cublasHandle), "Failed to destroy cublas handle:");
  handleCudaStatus(cudaStreamDestroy(stream1), "Failed to destroy cuda stream 1:");
}

void ElementWiseDivisionTest::SetUp() {

}

void ElementWiseDivisionTest::TearDown() {

}

TEST_F(ElementWiseDivisionTest, KernelSmallVector) {
  const int numberOfRows = 5;

  Container::PinnedHostVector* hostVectorNumerator = new Container::PinnedHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    (*hostVectorNumerator)(i) = i;
  }

  Container::PinnedHostVector* hostVectorDenomitor = new Container::PinnedHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    (*hostVectorDenomitor)(i) = (i + 3) * 10;
  }

  Container::DeviceVector* numeratorDeviceVector = hostToDeviceStream1.transferVector(hostVectorNumerator);
  Container::DeviceVector* denomitorDeviceVector = hostToDeviceStream1.transferVector(hostVectorDenomitor);
  Container::DeviceVector* resultDeviceVector = new Container::DeviceVector(numberOfRows);

  kernelWrapper.elementWiseDivision(*numeratorDeviceVector, *denomitorDeviceVector, *resultDeviceVector);

  Container::HostVector* resultHostVector = deviceToHostStream1.transferVector(resultDeviceVector);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error in ElemtWiseDivisionTest: ");

  ASSERT_EQ(numberOfRows, resultHostVector->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    PRECISION x = (*hostVectorNumerator)(i) / (*hostVectorDenomitor)(i);
    EXPECT_EQ(x, (*resultHostVector)(i));
  }

  delete hostVectorNumerator;
  delete hostVectorDenomitor;
  delete numeratorDeviceVector;
  delete denomitorDeviceVector;
  delete resultDeviceVector;
  delete resultHostVector;
}

TEST_F(ElementWiseDivisionTest, KernelException) {
  const int numberOfRows = 5;

  Container::DeviceVector* numeratorDeviceVector = new Container::DeviceVector(numberOfRows);
  Container::DeviceVector* denomitorDeviceVector = new Container::DeviceVector(numberOfRows + 1);
  Container::DeviceVector* resultDeviceVector = new Container::DeviceVector(numberOfRows - 1);

  EXPECT_THROW(kernelWrapper.elementWiseDivision(*numeratorDeviceVector, *denomitorDeviceVector, *resultDeviceVector),
      CudaException);

  delete numeratorDeviceVector;
  delete denomitorDeviceVector;
  delete resultDeviceVector;

  numeratorDeviceVector = new Container::DeviceVector(numberOfRows + 1);
  denomitorDeviceVector = new Container::DeviceVector(numberOfRows);
  resultDeviceVector = new Container::DeviceVector(numberOfRows);

  EXPECT_THROW(kernelWrapper.elementWiseDivision(*numeratorDeviceVector, *denomitorDeviceVector, *resultDeviceVector),
      CudaException);

  delete numeratorDeviceVector;
  delete denomitorDeviceVector;
  delete resultDeviceVector;

  numeratorDeviceVector = new Container::DeviceVector(numberOfRows);
  denomitorDeviceVector = new Container::DeviceVector(numberOfRows);
  resultDeviceVector = new Container::DeviceVector(numberOfRows + 1);

  EXPECT_THROW(kernelWrapper.elementWiseDivision(*numeratorDeviceVector, *denomitorDeviceVector, *resultDeviceVector),
      CudaException);

  delete numeratorDeviceVector;
  delete denomitorDeviceVector;
  delete resultDeviceVector;

  numeratorDeviceVector = new Container::DeviceVector(numberOfRows);
  denomitorDeviceVector = new Container::DeviceVector(numberOfRows + 1);
  resultDeviceVector = new Container::DeviceVector(numberOfRows);

  EXPECT_THROW(kernelWrapper.elementWiseDivision(*numeratorDeviceVector, *denomitorDeviceVector, *resultDeviceVector),
      CudaException);

  delete numeratorDeviceVector;
  delete denomitorDeviceVector;
  delete resultDeviceVector;

  numeratorDeviceVector = new Container::DeviceVector(numberOfRows - 1);
  denomitorDeviceVector = new Container::DeviceVector(numberOfRows - 1);
  resultDeviceVector = new Container::DeviceVector(numberOfRows);

  EXPECT_THROW(kernelWrapper.elementWiseDivision(*numeratorDeviceVector, *denomitorDeviceVector, *resultDeviceVector),
      CudaException);

  delete numeratorDeviceVector;
  delete denomitorDeviceVector;
  delete resultDeviceVector;

  numeratorDeviceVector = new Container::DeviceVector(numberOfRows - 1);
  denomitorDeviceVector = new Container::DeviceVector(numberOfRows);
  resultDeviceVector = new Container::DeviceVector(numberOfRows + 1);

  EXPECT_THROW(kernelWrapper.elementWiseDivision(*numeratorDeviceVector, *denomitorDeviceVector, *resultDeviceVector),
      CudaException);

  delete numeratorDeviceVector;
  delete denomitorDeviceVector;
  delete resultDeviceVector;

  numeratorDeviceVector = new Container::DeviceVector(numberOfRows);
  denomitorDeviceVector = new Container::DeviceVector(numberOfRows + 1);
  resultDeviceVector = new Container::DeviceVector(numberOfRows + 1);

  EXPECT_THROW(kernelWrapper.elementWiseDivision(*numeratorDeviceVector, *denomitorDeviceVector, *resultDeviceVector),
      CudaException);

  delete numeratorDeviceVector;
  delete denomitorDeviceVector;
  delete resultDeviceVector;
}

}
/* namespace CUDA */
} /* namespace CuEira */

