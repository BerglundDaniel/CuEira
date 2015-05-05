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
 * Test for testing the ApplyGeneticModel kernel
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ApplyGeneticModelTest: public ::testing::Test {
protected:
  ApplyGeneticModelTest();
  virtual ~ApplyGeneticModelTest();
  virtual void SetUp();
  virtual void TearDown();

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDevice;
  DeviceToHost deviceToHost;
  KernelWrapper kernelWrapper;
};

ApplyGeneticModelTest::ApplyGeneticModelTest() :
    device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDevice(*stream), deviceToHost(
        *stream), kernelWrapper(*stream) {

}

ApplyGeneticModelTest::~ApplyGeneticModelTest() {
  delete stream;
}

void ApplyGeneticModelTest::SetUp() {

}

void ApplyGeneticModelTest::TearDown() {

}

TEST_F(ApplyGeneticModelTest, KernelSmall) {
  const int numberOfRows = 10;
  const int snpToRisk[3] = {5, 7, 8};

  Container::PinnedHostVector fromHost(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    fromHost(i) = i % 3;
  }

  Container::DeviceVector* fromDevice = hostToDevice.transferVector(fromHost);
  Container::DeviceVector toDevice(numberOfRows);

  kernelWrapper.applyGeneticModel(snpToRisk, *fromDevice, toDevice);
  Container::PinnedHostVector* toHost = deviceToHost.transferVector(toDevice);

  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error in ApplyGeneticModelTest: ");

  ASSERT_EQ(numberOfRows, toHost->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(snpToRisk[fromHost(i)], (*toHost)(i));
  }

  delete fromDevice;
  delete toHost;
}

TEST_F(ApplyGeneticModelTest, KernelLarge) {
  const int numberOfRows = 1000;
  const int snpToRisk[3] = {1, 2, 3};

  Container::PinnedHostVector fromHost(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    fromHost(i) = i % 3;
  }

  Container::DeviceVector* fromDevice = hostToDevice.transferVector(fromHost);
  Container::DeviceVector toDevice(numberOfRows);

  kernelWrapper.applyGeneticModel(snpToRisk, *fromDevice, toDevice);
  Container::PinnedHostVector* toHost = deviceToHost.transferVector(toDevice);

  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error in ApplyGeneticModelTest: ");

  ASSERT_EQ(numberOfRows, toHost->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(snpToRisk[fromHost(i)], (*toHost)(i));
  }

  delete fromDevice;
  delete toHost;
}

#ifdef DEBUG
TEST_F(ApplyGeneticModelTest, KernelException){
  const int numberOfRows = 5;

  int snpToRisk[3] ={0, 1, 2};

  Container::DeviceVector from(numberOfRows);
  Container::DeviceVector to(numberOfRows);

  Container::DeviceVector fromW(numberOfRows + 1);
  Container::DeviceVector toW(numberOfRows - 1);

  EXPECT_THROW(kernelWrapper.applyGeneticModel(snpToRisk, from, toW), CudaException);
  EXPECT_THROW(kernelWrapper.applyGeneticModel(snpToRisk, fromW, to), CudaException);

}
#endif

}
/* namespace CUDA */
} /* namespace CuEira */
