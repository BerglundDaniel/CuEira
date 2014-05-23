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

namespace CuEira {
namespace CUDA {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class AbsoluteDifferenceTest: public ::testing::Test {
protected:
  AbsoluteDifferenceTest();
  virtual ~AbsoluteDifferenceTest();
  virtual void SetUp();
  virtual void TearDown();

  cublasStatus_t cublasStatus;
  cudaStream_t stream1;
  cublasHandle_t cublasHandle;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
  KernelWrapper kernelWrapper;
};

AbsoluteDifferenceTest::AbsoluteDifferenceTest() :
    cublasStatus(cublasCreate(&cublasHandle)), hostToDeviceStream1(HostToDevice(stream1)), deviceToHostStream1(
        DeviceToHost(stream1)), kernelWrapper(stream1,cublasHandle) {

  handleCublasStatus(cublasStatus, "Failed to create cublas handle:");
  handleCudaStatus(cudaStreamCreate(&stream1), "Failed to create cuda stream 1:");
  handleCublasStatus(cublasSetStream(cublasHandle, stream1), "Failed to set cuda stream:");
}

AbsoluteDifferenceTest::~AbsoluteDifferenceTest() {
  handleCublasStatus(cublasDestroy(cublasHandle), "Failed to destroy cublas handle:");
  handleCudaStatus(cudaStreamDestroy(stream1), "Failed to destroy cuda stream 1:");
}

void AbsoluteDifferenceTest::SetUp() {

}

void AbsoluteDifferenceTest::TearDown() {

}

TEST_F(AbsoluteDifferenceTest, KernelSmallVector) {
  const int numberOfRows = 5;

  Container::PinnedHostVector* hostVector1 = new Container::PinnedHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    (*hostVector1)(i) = i;
  }

  Container::PinnedHostVector* hostVector2 = new Container::PinnedHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    (*hostVector2)(i) = (i + 3) * 10;
  }

  Container::DeviceVector* deviceVector1 = hostToDeviceStream1.transferVector(hostVector1);
  Container::DeviceVector* deviceVector2 = hostToDeviceStream1.transferVector(hostVector2);
  Container::DeviceVector* resultDeviceVector = new Container::DeviceVector(numberOfRows);

  kernelWrapper.absoluteDifference(*deviceVector1, *deviceVector2, *resultDeviceVector);

  Container::HostVector* resultHostVector = deviceToHostStream1.transferVector(resultDeviceVector);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error in ElemtWiseDivisionTest: ");

  ASSERT_EQ(numberOfRows, resultHostVector->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    PRECISION x = std::abs((*hostVector1)(i) - (*hostVector2)(i));
    EXPECT_EQ(x, (*resultHostVector)(i));
  }

  delete hostVector1;
  delete hostVector2;
  delete deviceVector1;
  delete deviceVector2;
  delete resultDeviceVector;
  delete resultHostVector;
}

//TODO test exception

}
/* namespace CUDA */
} /* namespace CuEira */

