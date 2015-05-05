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
 * Test for testing the ApplyAdditiveModel kernel
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ApplyAdditiveModelTest: public ::testing::Test {
protected:
  ApplyAdditiveModelTest();
  virtual ~ApplyAdditiveModelTest();
  virtual void SetUp();
  virtual void TearDown();

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDevice;
  DeviceToHost deviceToHost;
  KernelWrapper kernelWrapper;
};

ApplyAdditiveModelTest::ApplyAdditiveModelTest() :
    device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDevice(*stream), deviceToHost(
        *stream), kernelWrapper(*stream) {

}

ApplyAdditiveModelTest::~ApplyAdditiveModelTest() {
  delete stream;
}

void ApplyAdditiveModelTest::SetUp() {

}

void ApplyAdditiveModelTest::TearDown() {

}

TEST_F(ApplyAdditiveModelTest, KernelSmall) {
  const int numberOfRows = 10;

  Container::PinnedHostVector vector1Host(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    vector1Host(i) = i % 3;
  }

  Container::PinnedHostVector vector2Host(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    if(i < numberOfRows / 2){
      vector2Host(i) = 0;
    }else{
      vector2Host(i) = 1;
    }
  }

  Container::DeviceVector* vector1Device = hostToDevice.transferVector(vector1Host);
  Container::DeviceVector* vector2Device = hostToDevice.transferVector(vector2Host);
  Container::DeviceVector interactionDevice(numberOfRows);

  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error in ApplyAdditiveModelTest: ");

  kernelWrapper.applyAdditiveModel(*vector1Device, *vector2Device, interactionDevice);

  Container::PinnedHostVector* vector1ResHost = deviceToHost.transferVector(*vector1Device);
  Container::PinnedHostVector* vector2ResHost = deviceToHost.transferVector(*vector2Device);
  Container::PinnedHostVector* interactionResHost = deviceToHost.transferVector(interactionDevice);

  ASSERT_EQ(numberOfRows, vector1ResHost->getNumberOfRows());
  ASSERT_EQ(numberOfRows, vector2ResHost->getNumberOfRows());
  ASSERT_EQ(numberOfRows, interactionResHost->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    int num = vector1Host(i) * vector2Host(i);
    EXPECT_EQ(num, (*interactionResHost)(i));

    if(num != 0){
      EXPECT_EQ(0, (*vector1ResHost)(i));
      EXPECT_EQ(0, (*vector2ResHost)(i));
    }else{
      EXPECT_EQ(vector1Host(i), (*vector1ResHost)(i));
      EXPECT_EQ(vector2Host(i), (*vector2ResHost)(i));
    }
  }

  delete vector1Device;
  delete vector2Device;
  delete vector1ResHost;
  delete vector2ResHost;
  delete interactionResHost;
}

TEST_F(ApplyAdditiveModelTest, KernelLarge) {
  const int numberOfRows = 1000;

  Container::PinnedHostVector vector1Host(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    vector1Host(i) = i % 3;
  }

  Container::PinnedHostVector vector2Host(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    if(i < numberOfRows / 2){
      vector2Host(i) = 0;
    }else{
      vector2Host(i) = 1;
    }
  }

  Container::DeviceVector* vector1Device = hostToDevice.transferVector(vector1Host);
  Container::DeviceVector* vector2Device = hostToDevice.transferVector(vector2Host);
  Container::DeviceVector interactionDevice(numberOfRows);

  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error in ApplyAdditiveModelTest: ");

  kernelWrapper.applyAdditiveModel(*vector1Device, *vector2Device, interactionDevice);

  Container::PinnedHostVector* vector1ResHost = deviceToHost.transferVector(*vector1Device);
  Container::PinnedHostVector* vector2ResHost = deviceToHost.transferVector(*vector2Device);
  Container::PinnedHostVector* interactionResHost = deviceToHost.transferVector(interactionDevice);

  ASSERT_EQ(numberOfRows, vector1ResHost->getNumberOfRows());
  ASSERT_EQ(numberOfRows, vector2ResHost->getNumberOfRows());
  ASSERT_EQ(numberOfRows, interactionResHost->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    int num = vector1Host(i) * vector2Host(i);
    EXPECT_EQ(num, (*interactionResHost)(i));

    if(num != 0){
      EXPECT_EQ(0, (*vector1ResHost)(i));
      EXPECT_EQ(0, (*vector2ResHost)(i));
    }else{
      EXPECT_EQ(vector1Host(i), (*vector1ResHost)(i));
      EXPECT_EQ(vector2Host(i), (*vector2ResHost)(i));
    }
  }

  delete vector1Device;
  delete vector2Device;
  delete vector1ResHost;
  delete vector2ResHost;
  delete interactionResHost;
}

#ifdef DEBUG
TEST_F(ApplyAdditiveModelTest, KernelException){
  const int numberOfRows = 5;

  Container::DeviceVector deviceVector1(numberOfRows);
  Container::DeviceVector deviceVector2(numberOfRows);
  Container::DeviceVector interactionDevice(numberOfRows);

  Container::DeviceVector deviceVectorW1(numberOfRows + 1);
  Container::DeviceVector deviceVectorW2(numberOfRows + 1);
  Container::DeviceVector interactionDeviceW(numberOfRows - 1);

  EXPECT_THROW(kernelWrapper.applyAdditiveModel(deviceVectorW1, deviceVector2, interactionDevice), CudaException);
  EXPECT_THROW(kernelWrapper.applyAdditiveModel(deviceVector1, deviceVectorW2, interactionDevice), CudaException);
  EXPECT_THROW(kernelWrapper.applyAdditiveModel(deviceVector1, deviceVector2, interactionDeviceW), CudaException);
}
#endif

}
/* namespace CUDA */
} /* namespace CuEira */
