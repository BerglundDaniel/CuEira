#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <sstream>
#include <math.h>
#include <stdlib.h>
#include <time.h>

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
 * Test for testing the VectorCopyIndexes kernel
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class VectorCopyIndexesTest: public ::testing::Test {
protected:
  VectorCopyIndexesTest();
  virtual ~VectorCopyIndexesTest();
  virtual void SetUp();
  virtual void TearDown();

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDevice;
  DeviceToHost deviceToHost;
  KernelWrapper kernelWrapper;
};

VectorCopyIndexesTest::VectorCopyIndexesTest() :
    device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDevice(*stream), deviceToHost(
        *stream), kernelWrapper(*stream) {

}

VectorCopyIndexesTest::~VectorCopyIndexesTest() {
  delete stream;
}

void VectorCopyIndexesTest::SetUp() {

}

void VectorCopyIndexesTest::TearDown() {

}

TEST_F(VectorCopyIndexesTest, KernelSmall) {
  const int numberOfRows = 10;
  srand(time(NULL));

  Container::PinnedHostVector indexesHost(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    indexesHost(i) = rand() % numberOfRows;
  }

  Container::PinnedHostVector fromHost(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    fromHost(i) = i;
  }

  Container::DeviceVector* indexesDevice = hostToDevice.transferVector(indexesHost);
  Container::DeviceVector* fromDevice = hostToDevice.transferVector(fromHost);
  Container::DeviceVector toDevice(numberOfRows);

  kernelWrapper.vectorCopyIndexes(*indexesDevice, *fromDevice, toDevice);
  Container::PinnedHostVector* toHost = deviceToHost.transferVector(toDevice);

  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error in VectorCopyIndexesTest: ");

  ASSERT_EQ(numberOfRows, toHost->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(fromHost(indexesHost(i)), (*toHost)(i));
  }

  delete indexesDevice;
  delete fromDevice;
  delete toHost;
}

TEST_F(VectorCopyIndexesTest, KernelLarge) {
  const int numberOfRows = 1000;
  srand(time(NULL));

  Container::PinnedHostVector indexesHost(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    indexesHost(i) = rand() % numberOfRows;
  }

  Container::PinnedHostVector fromHost(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    fromHost(i) = i;
  }

  Container::DeviceVector* indexesDevice = hostToDevice.transferVector(indexesHost);
  Container::DeviceVector* fromDevice = hostToDevice.transferVector(fromHost);
  Container::DeviceVector toDevice(numberOfRows);

  kernelWrapper.vectorCopyIndexes(*indexesDevice, *fromDevice, toDevice);
  Container::PinnedHostVector* toHost = deviceToHost.transferVector(toDevice);

  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error in VectorCopyIndexesTest: ");

  ASSERT_EQ(numberOfRows, toHost->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    EXPECT_EQ(fromHost(indexesHost(i)), (*toHost)(i));
  }

  delete indexesDevice;
  delete fromDevice;
  delete toHost;
}

#ifdef DEBUG
TEST_F(VectorCopyIndexesTest, KernelException){
  const int numberOfRows = 5;

  Container::DeviceVector indexes(numberOfRows);
  Container::DeviceVector from(numberOfRows);
  Container::DeviceVector to(numberOfRows);

  Container::DeviceVector indexesW(numberOfRows + 10);
  Container::DeviceVector fromW(numberOfRows + 1);
  Container::DeviceVector toW(numberOfRows - 1);

  EXPECT_THROW(kernelWrapper.vectorCopyIndexes(indexes, from, toW), CudaException);
  EXPECT_THROW(kernelWrapper.vectorCopyIndexes(indexes, fromW, to), CudaException);
  EXPECT_THROW(kernelWrapper.vectorCopyIndexes(indexesW, from, to), CudaException);
}
#endif

}
/* namespace CUDA */
} /* namespace CuEira */
