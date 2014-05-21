#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>

#include <CudaAdapter.cu>
#include <CudaException.h>
#include <CublasException.h>
#include <DimensionMismatch.h>

namespace CuEira {
namespace CUDA {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaAdapterTest: public ::testing::Test {
protected:
  CudaAdapterTest();
  virtual ~CudaAdapterTest();
  virtual void SetUp();
  virtual void TearDown();
};

CudaAdapterTest::CudaAdapterTest() {

}

CudaAdapterTest::~CudaAdapterTest() {

}

void CudaAdapterTest::SetUp() {

}

void CudaAdapterTest::TearDown() {

}

TEST_F(CudaAdapterTest, CublasGetErrorString) {
  EXPECT_EQ("CUBLAS_STATUS_SUCCESS", cublasGetErrorString(CUBLAS_STATUS_SUCCESS));
  EXPECT_EQ("CUBLAS_STATUS_NOT_INITIALIZED", cublasGetErrorString(CUBLAS_STATUS_NOT_INITIALIZED));
  EXPECT_EQ("CUBLAS_STATUS_ALLOC_FAILED", cublasGetErrorString(CUBLAS_STATUS_ALLOC_FAILED));
  EXPECT_EQ("CUBLAS_STATUS_INVALID_VALUE", cublasGetErrorString(CUBLAS_STATUS_INVALID_VALUE));
  EXPECT_EQ("CUBLAS_STATUS_ARCH_MISMATCH", cublasGetErrorString(CUBLAS_STATUS_ARCH_MISMATCH));
  EXPECT_EQ("CUBLAS_STATUS_MAPPING_ERROR", cublasGetErrorString(CUBLAS_STATUS_MAPPING_ERROR));
  EXPECT_EQ("CUBLAS_STATUS_EXECUTION_FAILED", cublasGetErrorString(CUBLAS_STATUS_EXECUTION_FAILED));
  EXPECT_EQ("CUBLAS_STATUS_INTERNAL_ERROR", cublasGetErrorString(CUBLAS_STATUS_INTERNAL_ERROR));
}

TEST_F(CudaAdapterTest, HandleCudaStatus) {
  handleCudaStatus(cudaSuccess, "test");
  EXPECT_THROW(handleCudaStatus(cudaErrorMissingConfiguration, "test"), CudaException);
  EXPECT_THROW(handleCudaStatus(cudaErrorNoDevice, "test"), CudaException);
  EXPECT_THROW(handleCudaStatus(cudaErrorIncompatibleDriverContext, "test"), CudaException);
}

TEST_F(CudaAdapterTest, HandleCublasStatus) {
  handleCublasStatus(CUBLAS_STATUS_SUCCESS, "test");
  EXPECT_THROW(handleCublasStatus(CUBLAS_STATUS_NOT_INITIALIZED, "test"), CublasException);
  EXPECT_THROW(handleCublasStatus(CUBLAS_STATUS_ALLOC_FAILED, "test"), CublasException);
  EXPECT_THROW(handleCublasStatus(CUBLAS_STATUS_INVALID_VALUE, "test"), CublasException);
  EXPECT_THROW(handleCublasStatus(CUBLAS_STATUS_ARCH_MISMATCH, "test"), CublasException);
  EXPECT_THROW(handleCublasStatus(CUBLAS_STATUS_MAPPING_ERROR, "test"), CublasException);
  EXPECT_THROW(handleCublasStatus(CUBLAS_STATUS_EXECUTION_FAILED, "test"), CublasException);
  EXPECT_THROW(handleCublasStatus(CUBLAS_STATUS_INTERNAL_ERROR, "test"), CublasException);
}

}
/* namespace CUDA */
} /* namespace CuEira */

