#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>
#include <CudaException.h>
#include <DeviceMock.h>
#include <Stream.h>

namespace CuEira {
namespace CUDA {

/**
 * Test for testing
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class StreamTest: public ::testing::Test {
protected:
  StreamTest();
  virtual ~StreamTest();
  virtual void SetUp();
  virtual void TearDown();
};

StreamTest::StreamTest() {

}

StreamTest::~StreamTest() {

}

void StreamTest::SetUp() {

}

void StreamTest::TearDown() {

}

TEST_F(StreamTest, GetCuda) {
  cudaStream_t* cudaStream = new cudaStream_t();
  cublasHandle_t* cublasHandle = new cublasHandle_t();

  handleCublasStatus(cublasCreate(cublasHandle), "Failed to create new cublas handle:");
  handleCudaStatus(cudaStreamCreate(cudaStream), "Failed to create new cuda stream:");
  handleCublasStatus(cublasSetStream(*cublasHandle, *cudaStream), "Failed to set cuda stream for cublas handle:");

  DeviceMock deviceMock;

  Stream stream(deviceMock, cudaStream, cublasHandle);

  EXPECT_EQ(cudaStream, &stream.getCudaStream());
  EXPECT_EQ(cublasHandler, &stream.getCublasHandle());
}

TEST_F(StreamTest, GetDevice) {
  cudaStream_t* cudaStream = new cudaStream_t();
  cublasHandle_t* cublasHandle = new cublasHandle_t();

  handleCublasStatus(cublasCreate(cublasHandle), "Failed to create new cublas handle:");
  handleCudaStatus(cudaStreamCreate(cudaStream), "Failed to create new cuda stream:");
  handleCublasStatus(cublasSetStream(*cublasHandle, *cudaStream), "Failed to set cuda stream for cublas handle:");

  DeviceMock deviceMock;

  Stream stream(deviceMock, cudaStream, cublasHandle);

  EXPECT_EQ(&deviceMock, &stream.getAssociatedDevice());
}

}
/* namespace CUDA */
} /* namespace CuEira*/

