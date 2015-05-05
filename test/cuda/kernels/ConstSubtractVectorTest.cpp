#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <sstream>
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
#include <Device.h>
#include <Stream.h>
#include <StreamFactory.h>

using ::testing::Ge;
using ::testing::Le;

namespace CuEira {
namespace CUDA {

/**
 * Test for testing the ConstSubtractVector kernel
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ConstSubtractVectorTest: public ::testing::Test {
protected:
  ConstSubtractVectorTest();
  virtual ~ConstSubtractVectorTest();
  virtual void SetUp();
  virtual void TearDown();

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDevice;
  DeviceToHost deviceToHost;
  KernelWrapper kernelWrapper;
};

ConstSubtractVectorTest::ConstSubtractVectorTest() :
    device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDevice(*stream), deviceToHost(
        *stream), kernelWrapper(*stream) {

}

ConstSubtractVectorTest::~ConstSubtractVectorTest() {
  delete stream;
}

void ConstSubtractVectorTest::SetUp() {

}

void ConstSubtractVectorTest::TearDown() {

}

TEST_F(ConstSubtractVectorTest, KernelSmall) {
  const int numberOfRows = 10;
  const int c = 5;

  Container::PinnedHostVector vectorHost(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    vectorHost(i) = i % 20;
  }

  Container::DeviceVector* fromDevice = hostToDevice.transferVector(vectorHost);

  kernelWrapper.constSubtractVector(c, *vectorDevice);
  Container::PinnedHostVector* resHost = deviceToHost.transferVector(*vector1Device);

  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error in ConstSubtractVectorTest: ");

  ASSERT_EQ(numberOfRows, resHost->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(c - vectorHost(i), (*resHost)(i));
  }

  delete vectorDevice;
  delete resHost;
}

TEST_F(ConstSubtractVectorTest, KernelLarge) {
  const int numberOfRows = 1000;
  const int c = 5;

  Container::PinnedHostVector vectorHost(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    vectorHost(i) = i % 20;
  }

  Container::DeviceVector* fromDevice = hostToDevice.transferVector(vectorHost);

  kernelWrapper.constSubtractVector(c, *vectorDevice);
  Container::PinnedHostVector* resHost = deviceToHost.transferVector(*vector1Device);

  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error in ConstSubtractVectorTest: ");

  ASSERT_EQ(numberOfRows, resHost->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(c - vectorHost(i), (*resHost)(i));
  }

  delete vectorDevice;
  delete resHost;
}

}
/* namespace CUDA */
} /* namespace CuEira */
