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
class ElementWiseAdditionTest: public ::testing::Test {
protected:
  ElementWiseAdditionTest();
  virtual ~ElementWiseAdditionTest();
  virtual void SetUp();
  virtual void TearDown();

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
  KernelWrapper kernelWrapper;
};

ElementWiseAdditionTest::ElementWiseAdditionTest() :
    device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDeviceStream1(*stream), deviceToHostStream1(
        *stream), kernelWrapper(*stream) {
}

ElementWiseAdditionTest::~ElementWiseAdditionTest() {
  delete stream;
}

void ElementWiseAdditionTest::SetUp() {

}

void ElementWiseAdditionTest::TearDown() {

}

TEST_F(ElementWiseAdditionTest, KernelSmallVector) {
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

  kernelWrapper.elementWiseAddition(*deviceVector1, *deviceVector2, *resultDeviceVector);

  Container::HostVector* resultHostVector = deviceToHostStream1.transferVector(resultDeviceVector);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error in ElementWiseAdditionTest: ");

  ASSERT_EQ(numberOfRows, resultHostVector->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    PRECISION x = (*hostVector1)(i) + (*hostVector2)(i);
    EXPECT_EQ(x, (*resultHostVector)(i));
  }

  delete hostVector1;
  delete hostVector2;
  delete deviceVector1;
  delete deviceVector2;
  delete resultDeviceVector;
  delete resultHostVector;
}

TEST_F(ElementWiseAdditionTest, KernelException) {
  const int numberOfRows = 5;

  Container::DeviceVector* deviceVector1 = new Container::DeviceVector(numberOfRows + 1);
  Container::DeviceVector* deviceVector2 = new Container::DeviceVector(numberOfRows);
  Container::DeviceVector* resultDeviceVector = new Container::DeviceVector(numberOfRows);

  EXPECT_THROW(kernelWrapper.elementWiseAddition(*deviceVector1, *deviceVector2, *resultDeviceVector), CudaException);

  delete deviceVector1;
  delete deviceVector2;
  delete resultDeviceVector;

  deviceVector1 = new Container::DeviceVector(numberOfRows + 1);
  deviceVector2 = new Container::DeviceVector(numberOfRows + 1);
  resultDeviceVector = new Container::DeviceVector(numberOfRows);

  EXPECT_THROW(kernelWrapper.elementWiseAddition(*deviceVector1, *deviceVector2, *resultDeviceVector), CudaException);

  delete deviceVector1;
  delete deviceVector2;
  delete resultDeviceVector;

  deviceVector1 = new Container::DeviceVector(numberOfRows + 1);
  deviceVector2 = new Container::DeviceVector(numberOfRows);
  resultDeviceVector = new Container::DeviceVector(numberOfRows + 1);

  EXPECT_THROW(kernelWrapper.elementWiseAddition(*deviceVector1, *deviceVector2, *resultDeviceVector), CudaException);

  delete deviceVector1;
  delete deviceVector2;
  delete resultDeviceVector;

  deviceVector1 = new Container::DeviceVector(numberOfRows);
  deviceVector2 = new Container::DeviceVector(numberOfRows + 1);
  resultDeviceVector = new Container::DeviceVector(numberOfRows + 1);

  EXPECT_THROW(kernelWrapper.elementWiseAddition(*deviceVector1, *deviceVector2, *resultDeviceVector), CudaException);

  delete deviceVector1;
  delete deviceVector2;
  delete resultDeviceVector;

  deviceVector1 = new Container::DeviceVector(numberOfRows);
  deviceVector2 = new Container::DeviceVector(numberOfRows);
  resultDeviceVector = new Container::DeviceVector(numberOfRows + 1);

  EXPECT_THROW(kernelWrapper.elementWiseAddition(*deviceVector1, *deviceVector2, *resultDeviceVector), CudaException);

  delete deviceVector1;
  delete deviceVector2;
  delete resultDeviceVector;

  deviceVector1 = new Container::DeviceVector(numberOfRows - 1);
  deviceVector2 = new Container::DeviceVector(numberOfRows - 1);
  resultDeviceVector = new Container::DeviceVector(numberOfRows);

  EXPECT_THROW(kernelWrapper.elementWiseAddition(*deviceVector1, *deviceVector2, *resultDeviceVector), CudaException);

  delete deviceVector1;
  delete deviceVector2;
  delete resultDeviceVector;
}

}
/* namespace CUDA */
} /* namespace CuEira */

