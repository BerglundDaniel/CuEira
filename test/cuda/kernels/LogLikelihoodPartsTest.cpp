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
class LogLikelihoodPartsTest: public ::testing::Test {
protected:
  LogLikelihoodPartsTest();
  virtual ~LogLikelihoodPartsTest();
  virtual void SetUp();
  virtual void TearDown();

  cublasStatus_t cublasStatus;
  cudaStream_t stream1;
  cublasHandle_t cublasHandle;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
  KernelWrapper kernelWrapper;
};

LogLikelihoodPartsTest::LogLikelihoodPartsTest() :
    cublasStatus(cublasCreate(&cublasHandle)), hostToDeviceStream1(HostToDevice(stream1)), deviceToHostStream1(
        DeviceToHost(stream1)), kernelWrapper(stream1) {

  handleCublasStatus(cublasStatus, "Failed to create cublas handle:");
  handleCudaStatus(cudaStreamCreate(&stream1), "Failed to create cuda stream 1:");
  handleCublasStatus(cublasSetStream(cublasHandle, stream1), "Failed to set cuda stream:");
}

LogLikelihoodPartsTest::~LogLikelihoodPartsTest() {
  handleCublasStatus(cublasDestroy(cublasHandle), "Failed to destroy cublas handle:");
  handleCudaStatus(cudaStreamDestroy(stream1), "Failed to destroy cuda stream 1:");
}

void LogLikelihoodPartsTest::SetUp() {

}

void LogLikelihoodPartsTest::TearDown() {

}

TEST_F(LogLikelihoodPartsTest, KernelSmallVector) {
  const int numberOfRows = 5;

  Container::PinnedHostVector* hostVectorOutcomes = new Container::PinnedHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    (*hostVectorOutcomes)(i) = i;
  }

  Container::PinnedHostVector* hostVectorProbabilites = new Container::PinnedHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    (*hostVectorProbabilites)(i) = i / 10;
  }

  Container::DeviceVector* outcomesDeviceVector = hostToDeviceStream1.transferVector(hostVectorOutcomes);
  Container::DeviceVector* probabilitesDeviceVector = hostToDeviceStream1.transferVector(hostVectorProbabilites);
  Container::DeviceVector* resultDeviceVector = new Container::DeviceVector(numberOfRows);

  kernelWrapper.logLikelihoodParts(*outcomesDeviceVector, *probabilitesDeviceVector, *resultDeviceVector);

  Container::HostVector* resultHostVector = deviceToHostStream1.transferVector(resultDeviceVector);
  cudaStreamSynchronize(stream1);
  handleCudaStatus(cudaGetLastError(), "Error in ElemtWiseDivisionTest: ");

  ASSERT_EQ(numberOfRows, resultHostVector->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    PRECISION x = (*hostVectorOutcomes)(i) * std::log((*hostVectorProbabilites)(i))
        + (1 - (*hostVectorOutcomes)(i)) * std::log(1 - (*hostVectorProbabilites)(i));
    EXPECT_EQ(x, (*resultHostVector)(i));
  }

  delete hostVectorOutcomes;
  delete hostVectorProbabilites;
  delete outcomesDeviceVector;
  delete probabilitesDeviceVector;
  delete resultDeviceVector;
  delete resultHostVector;
}

//TODO test exception

}
/* namespace CUDA */
} /* namespace CuEira */

