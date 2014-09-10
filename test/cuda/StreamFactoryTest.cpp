#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>
#include <CudaException.h>
#include <DeviceMock.h>
#include <Stream.h>
#include <StreamFactory.h>

namespace CuEira {
namespace CUDA {

/**
 * Test for testing
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class StreamFactoryTest: public ::testing::Test {
protected:
  StreamFactoryTest();
  virtual ~StreamFactoryTest();
  virtual void SetUp();
  virtual void TearDown();
};

StreamFactoryTest::StreamFactoryTest() {

}

StreamFactoryTest::~StreamFactoryTest() {

}

void StreamFactoryTest::SetUp() {

}

void StreamFactoryTest::TearDown() {

}

TEST_F(StreamFactoryTest, ConstructStream) {
  StreamFactory streamFactory;
  DeviceMock deviceMock;

  Stream* stream = streamFactory.constructStream(deviceMock);

  ASSERT_EQ(deviceMock, stream->getAssociatedDevice());

  delete stream;
}

}
/* namespace CUDA */
} /* namespace CuEira*/

