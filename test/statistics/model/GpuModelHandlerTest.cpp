#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <ostream>

#include <HostVector.h>
#include <HostMatrix.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

namespace CuEira {
namespace Model {

/**
 * Test for testing GpuModelHandler
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class GpuModelHandlerTest: public ::testing::Test {
protected:
  GpuModelHandlerTest();
  virtual ~GpuModelHandlerTest();
  virtual void SetUp();
  virtual void TearDown();

};

GpuModelHandlerTest::GpuModelHandlerTest() {

}

GpuModelHandlerTest::~GpuModelHandlerTest() {

}

void GpuModelHandlerTest::SetUp() {

}

void GpuModelHandlerTest::TearDown() {

}

TEST_F(GpuModelHandlerTest, Construct) {

}

}
/* namespace Model */
} /* namespace CuEira */
