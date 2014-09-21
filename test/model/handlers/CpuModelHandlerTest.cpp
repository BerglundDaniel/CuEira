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
 * Test for testing CpuModelHandler
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuModelHandlerTest: public ::testing::Test {
protected:
  CpuModelHandlerTest();
  virtual ~CpuModelHandlerTest();
  virtual void SetUp();
  virtual void TearDown();

};

CpuModelHandlerTest::CpuModelHandlerTest() {

}

CpuModelHandlerTest::~CpuModelHandlerTest() {

}

void CpuModelHandlerTest::SetUp() {

}

void CpuModelHandlerTest::TearDown() {

}

TEST_F(CpuModelHandlerTest, Construct) {

}

}
/* namespace Model */
} /* namespace CuEira */
