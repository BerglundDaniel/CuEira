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
#include <Device.h>
#include <Stream.h>
#include <StreamFactory.h>

using ::testing::Ge;
using ::testing::Le;

namespace CuEira {
namespace CUDA {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LogisticTransformTest: public ::testing::Test {
protected:
  LogisticTransformTest();
  virtual ~LogisticTransformTest();
  virtual void SetUp();
  virtual void TearDown();

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
  KernelWrapper kernelWrapper;
};

LogisticTransformTest::LogisticTransformTest() :
        device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDeviceStream1(*stream), deviceToHostStream1(
            *stream), kernelWrapper(*stream) {
}

LogisticTransformTest::~LogisticTransformTest() {
  delete stream;
}

void LogisticTransformTest::SetUp() {

}

void LogisticTransformTest::TearDown() {

}

TEST_F(LogisticTransformTest, KernelSmallVector) {
  const int numberOfRows = 5;
  double e = 10e-5;

  Container::PinnedHostVector* hostVectorFrom = new Container::PinnedHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    (*hostVectorFrom)(i) = i / 10;
  }

  Container::DeviceVector* logitDeviceVector = hostToDeviceStream1.transferVector(hostVectorFrom);
  Container::DeviceVector* probDeviceVector = new Container::DeviceVector(numberOfRows);

  kernelWrapper.logisticTransform(*logitDeviceVector, *probDeviceVector);

  Container::HostVector* resultHostVector = deviceToHostStream1.transferVector(probDeviceVector);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error in LogisticTransform test: ");

  ASSERT_EQ(numberOfRows, resultHostVector->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    PRECISION x = i / 10;
    x = exp(x) / (1 + exp(x));
    double l = x - e;
    double h = x + e;

    EXPECT_THAT((*resultHostVector)(i), Ge(l));
    EXPECT_THAT((*resultHostVector)(i), Le(h));
  }

  delete hostVectorFrom;
  delete logitDeviceVector;
  delete probDeviceVector;
  delete resultHostVector;
}

TEST_F(LogisticTransformTest, KernelException) {
  const int numberOfRows = 5;

  Container::DeviceVector* logitDeviceVector = new Container::DeviceVector(numberOfRows + 1);
  Container::DeviceVector* probDeviceVector = new Container::DeviceVector(numberOfRows);

  EXPECT_THROW(kernelWrapper.logisticTransform(*logitDeviceVector, *probDeviceVector), CudaException);

  delete logitDeviceVector;
  delete probDeviceVector;

  logitDeviceVector = new Container::DeviceVector(numberOfRows);
  probDeviceVector = new Container::DeviceVector(numberOfRows + 1);

  EXPECT_THROW(kernelWrapper.logisticTransform(*logitDeviceVector, *probDeviceVector), CudaException);

  delete logitDeviceVector;
  delete probDeviceVector;

  logitDeviceVector = new Container::DeviceVector(numberOfRows);
  probDeviceVector = new Container::DeviceVector(numberOfRows - 1);

  EXPECT_THROW(kernelWrapper.logisticTransform(*logitDeviceVector, *probDeviceVector), CudaException);

  delete logitDeviceVector;
  delete probDeviceVector;
}

TEST_F(LogisticTransformTest, KernelLargeVector) {
  const int numberOfRows = 10000;
  double e = 10e-5;

  Container::PinnedHostVector* hostVectorFrom = new Container::PinnedHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    (*hostVectorFrom)(i) = (i % 10) / 10;
  }

  Container::DeviceVector* logitDeviceVector = hostToDeviceStream1.transferVector(hostVectorFrom);
  Container::DeviceVector* probDeviceVector = new Container::DeviceVector(numberOfRows);

  kernelWrapper.logisticTransform(*logitDeviceVector, *probDeviceVector);

  Container::HostVector* resultHostVector = deviceToHostStream1.transferVector(probDeviceVector);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error in LogisticTransform test: ");

  ASSERT_EQ(numberOfRows, resultHostVector->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    PRECISION x = (*hostVectorFrom)(i);
    x = exp(x) / (1 + exp(x));
    double l = x - e;
    double h = x + e;

    EXPECT_THAT((*resultHostVector)(i), Ge(l));
    EXPECT_THAT((*resultHostVector)(i), Le(h));
  }

  delete hostVectorFrom;
  delete logitDeviceVector;
  delete probDeviceVector;
  delete resultHostVector;
}

TEST_F(LogisticTransformTest, KernelHugeVector) {
  const int numberOfRows = 100000;
  double e = 10e-5;

  Container::PinnedHostVector* hostVectorFrom = new Container::PinnedHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    (*hostVectorFrom)(i) = (i % 10) / 10;
  }

  Container::DeviceVector* logitDeviceVector = hostToDeviceStream1.transferVector(hostVectorFrom);
  Container::DeviceVector* probDeviceVector = new Container::DeviceVector(numberOfRows);

  kernelWrapper.logisticTransform(*logitDeviceVector, *probDeviceVector);

  Container::HostVector* resultHostVector = deviceToHostStream1.transferVector(probDeviceVector);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error in LogisticTransform test: ");

  ASSERT_EQ(numberOfRows, resultHostVector->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    PRECISION x = (*hostVectorFrom)(i);
    x = exp(x) / (1 + exp(x));
    double l = x - e;
    double h = x + e;

    EXPECT_THAT((*resultHostVector)(i), Ge(l));
    EXPECT_THAT((*resultHostVector)(i), Le(h));
  }

  delete hostVectorFrom;
  delete logitDeviceVector;
  delete probDeviceVector;
  delete resultHostVector;
}

}
/* namespace CUDA */
} /* namespace CuEira */

