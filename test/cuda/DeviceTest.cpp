#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>
#include <CudaException.h>
#include <Device.h>

namespace CuEira {
namespace CUDA {

/**
 * Test for testing
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DeviceTest: public ::testing::Test {
protected:
  DeviceTest();
  virtual ~DeviceTest();
  virtual void SetUp();
  virtual void TearDown();
};

DeviceTest::DeviceTest() {

}

DeviceTest::~DeviceTest() {

}

void DeviceTest::SetUp() {

}

void DeviceTest::TearDown() {

}

TEST_F(DeviceTest, SetGetDeviceSimple) {
  Device device(0, nullptr);

  device.setActiveDevice();
  ASSERT_TRUE(device.isActive());
}

TEST_F(DeviceTest, SetGetDeviceMult) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if(deviceCount < 2){
    std::cerr << "Skipping test for multiple device since only one device was detected." << std::endl;
    return;
  }

  Device device0(0, nullptr);
  Device device1(1, nullptr);

  device1.setActiveDevice();
  ASSERT_TRUE(device1.isActive());
  ASSERT_FALSE(device0.isActive());

  device0.setActiveDevice();
  ASSERT_TRUE(device0.isActive());
  ASSERT_FALSE(device1.isActive());
}

TEST_F(DeviceTest, GetOutcomes) {
  const int size = 5;
  Container::DeviceVector* outcomesDevice = new Container::DeviceVector(size);

  Device device(0, outcomesDevice);

  EXPECT_EQ(outcomesDevice, &device.getOutcomes());
}

}
/* namespace CUDA */
} /* namespace CuEira*/

