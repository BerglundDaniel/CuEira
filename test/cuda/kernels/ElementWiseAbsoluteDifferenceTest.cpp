#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

namespace CuEira {
namespace CUDA {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ElementWiseAbsoluteDifferenceTest: public ::testing::Test {
protected:
  ElementWiseAbsoluteDifferenceTest();
  virtual ~ElementWiseAbsoluteDifferenceTest();
  virtual void SetUp();
  virtual void TearDown();

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
  KernelWrapper kernelWrapper;
};

ElementWiseAbsoluteDifferenceTest::ElementWiseAbsoluteDifferenceTest() :
    device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDeviceStream1(*stream), deviceToHostStream1(
        *stream), kernelWrapper(*stream) {
}

ElementWiseAbsoluteDifferenceTest::~ElementWiseAbsoluteDifferenceTest() {
  delete stream;
}

void ElementWiseAbsoluteDifferenceTest::SetUp() {

}

void ElementWiseAbsoluteDifferenceTest::TearDown() {

}

TEST_F(ElementWiseAbsoluteDifferenceTest, KernelSmallVector) {
  const int numberOfRows = 5;

  Container::PinnedHostVector* hostVector1 = new Container::PinnedHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    (*hostVector1)(i) = i;
  }

  Container::PinnedHostVector* hostVector2 = new Container::PinnedHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    (*hostVector2)(i) = (i + 3) * 10;
  }

  Container::DeviceVector* deviceVector1 = hostToDeviceStream1.transferVector(*hostVector1);
  Container::DeviceVector* deviceVector2 = hostToDeviceStream1.transferVector(*hostVector2);
  Container::DeviceVector* resultDeviceVector = new Container::DeviceVector(numberOfRows);

  kernelWrapper.elementWiseAbsoluteDifference(*deviceVector1, *deviceVector2, *resultDeviceVector);

  Container::HostVector* resultHostVector = deviceToHostStream1.transferVector(*resultDeviceVector);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error in ElementWiseAbsoluteDifferenceTest: ");

  ASSERT_EQ(numberOfRows, resultHostVector->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    PRECISION x = fabs((*hostVector1)(i) - (*hostVector2)(i));
    EXPECT_EQ(x, (*resultHostVector)(i));
  }

  delete hostVector1;
  delete hostVector2;
  delete deviceVector1;
  delete deviceVector2;
  delete resultDeviceVector;
  delete resultHostVector;
}

TEST_F(ElementWiseAbsoluteDifferenceTest, KernelException) {
  const int numberOfRows = 5;

  Container::DeviceVector* deviceVector1 = new Container::DeviceVector(numberOfRows + 1);
  Container::DeviceVector* deviceVector2 = new Container::DeviceVector(numberOfRows);
  Container::DeviceVector* resultDeviceVector = new Container::DeviceVector(numberOfRows);

  EXPECT_THROW(kernelWrapper.elementWiseAbsoluteDifference(*deviceVector1, *deviceVector2, *resultDeviceVector),
      CudaException);

  delete deviceVector1;
  delete deviceVector2;
  delete resultDeviceVector;

  deviceVector1 = new Container::DeviceVector(numberOfRows + 1);
  deviceVector2 = new Container::DeviceVector(numberOfRows + 1);
  resultDeviceVector = new Container::DeviceVector(numberOfRows);

  EXPECT_THROW(kernelWrapper.elementWiseAbsoluteDifference(*deviceVector1, *deviceVector2, *resultDeviceVector),
      CudaException);

  delete deviceVector1;
  delete deviceVector2;
  delete resultDeviceVector;

  deviceVector1 = new Container::DeviceVector(numberOfRows + 1);
  deviceVector2 = new Container::DeviceVector(numberOfRows);
  resultDeviceVector = new Container::DeviceVector(numberOfRows + 1);

  EXPECT_THROW(kernelWrapper.elementWiseAbsoluteDifference(*deviceVector1, *deviceVector2, *resultDeviceVector),
      CudaException);

  delete deviceVector1;
  delete deviceVector2;
  delete resultDeviceVector;

  deviceVector1 = new Container::DeviceVector(numberOfRows);
  deviceVector2 = new Container::DeviceVector(numberOfRows + 1);
  resultDeviceVector = new Container::DeviceVector(numberOfRows + 1);

  EXPECT_THROW(kernelWrapper.elementWiseAbsoluteDifference(*deviceVector1, *deviceVector2, *resultDeviceVector),
      CudaException);

  delete deviceVector1;
  delete deviceVector2;
  delete resultDeviceVector;

  deviceVector1 = new Container::DeviceVector(numberOfRows);
  deviceVector2 = new Container::DeviceVector(numberOfRows);
  resultDeviceVector = new Container::DeviceVector(numberOfRows + 1);

  EXPECT_THROW(kernelWrapper.elementWiseAbsoluteDifference(*deviceVector1, *deviceVector2, *resultDeviceVector),
      CudaException);

  delete deviceVector1;
  delete deviceVector2;
  delete resultDeviceVector;

  deviceVector1 = new Container::DeviceVector(numberOfRows - 1);
  deviceVector2 = new Container::DeviceVector(numberOfRows - 1);
  resultDeviceVector = new Container::DeviceVector(numberOfRows);

  EXPECT_THROW(kernelWrapper.elementWiseAbsoluteDifference(*deviceVector1, *deviceVector2, *resultDeviceVector),
      CudaException);

  delete deviceVector1;
  delete deviceVector2;
  delete resultDeviceVector;
}

}
/* namespace CUDA */
} /* namespace CuEira */

