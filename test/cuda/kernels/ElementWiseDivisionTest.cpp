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
class ElemtWiseDivisionTest: public ::testing::Test {
protected:
  ElemtWiseDivisionTest();
  virtual ~ElemtWiseDivisionTest();
  virtual void SetUp();
  virtual void TearDown();

  cublasStatus_t cublasStatus;
  cudaStream_t stream1;
  cublasHandle_t cublasHandle;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
  KernelWrapper kernelWrapper;
};

ElemtWiseDivisionTest::ElemtWiseDivisionTest() :
    cublasStatus(cublasCreate(&cublasHandle)), hostToDeviceStream1(HostToDevice(stream1)), deviceToHostStream1(
        DeviceToHost(stream1)), kernelWrapper(stream1, cublasHandle) {

  handleCublasStatus(cublasStatus, "Failed to create cublas handle:");
  handleCudaStatus(cudaStreamCreate(&stream1), "Failed to create cuda stream 1:");
  handleCublasStatus(cublasSetStream(cublasHandle, stream1), "Failed to set cuda stream:");
}

ElemtWiseDivisionTest::~ElemtWiseDivisionTest() {
  handleCublasStatus(cublasDestroy(cublasHandle), "Failed to destroy cublas handle:");
  handleCudaStatus(cudaStreamDestroy(stream1), "Failed to destroy cuda stream 1:");
}

void ElemtWiseDivisionTest::SetUp() {

}

void ElemtWiseDivisionTest::TearDown() {

}

TEST_F(ElemtWiseDivisionTest, KernelSmallVector) {
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

//TODO test exception

}
/* namespace CUDA */
} /* namespace CuEira */

