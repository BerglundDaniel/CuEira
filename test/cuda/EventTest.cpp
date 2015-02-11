#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>
#include <CudaException.h>
#include <DeviceMock.h>
#include <Stream.h>
#include <Event.h>
#include <StreamFactory.h>

using testing::Ge;
using testing::Le;

namespace CuEira {
namespace CUDA {

class EventTest: public ::testing::Test {
protected:
  EventTest();
  virtual ~EventTest();
  virtual void SetUp();
  virtual void TearDown();
};

EventTest::EventTest() {

}

EventTest::~EventTest() {

}

void EventTest::SetUp() {

}

void EventTest::TearDown() {

}

TEST_F(EventTest, EventDifference) {
  const double e = 1e-2;
  StreamFactory streamFactory;

  Device device(0);
  device.setActiveDevice();
  ASSERT_TRUE(device.isActive());

  Stream* stream = streamFactory.constructStream(device);

  Event before(*stream);

  //TODO a kernel here that waits a while

  Event after(*stream);

  float diff = after - before;

  const float realDiff = 1;
  float l = realDiff - e;
  float h = realDiff + e;
  EXPECT_THAT(diff, Ge(l));
  EXPECT_THAT(diff, Le(h));

  delete stream;
}

}
/* namespace CUDA */
} /* namespace CuEira*/
