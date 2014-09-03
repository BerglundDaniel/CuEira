#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <ostream>

#include <HostVector.h>
#include <HostMatrix.h>
#include <DataHandler.h>
#include <LogisticRegressionMock.h>
#include <LogisticRegressionConfigurationMock.h>
#include <LogisticRegressionResultMock.h>
#include <SNP.h>
#include <Id.h>
#include <EnvironmentFactor.h>
#include <GpuModelHandler.h>
#include <ModelHandler.h>
#include <Statistics.h>
#include <StatisticsFactoryMock.h>
#include <PinnedHostMatrix.h>
#include <PinnedHostVector.h>
#include <DeviceMatrix.h>
#include <DeviceVector.h>
#include <HostToDevice.h>
#include <DeviceToHost.h>

using testing::Return;
using testing::ReturnRef;
using testing::_;
using testing::Ge;
using testing::Eq;
using testing::Le;
using testing::ByRef;
using testing::InSequence;

namespace CuEira {
namespace Model {

/**
 * Integration test for GpuModelHandler, LogisticRegressionConfiguration, LogisticRegression, LogisticRegressionResult, Statistics
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class GpuModelHandlerLRIntegrationTest: public ::testing::Test {
protected:
  GpuModelHandlerLRIntegrationTest();
  virtual ~GpuModelHandlerLRIntegrationTest();
  virtual void SetUp();
  virtual void TearDown();

};

GpuModelHandlerLRIntegrationTest::GpuModelHandlerLRIntegrationTest() {

}

GpuModelHandlerLRIntegrationTest::~GpuModelHandlerLRIntegrationTest() {

}

void GpuModelHandlerLRIntegrationTest::SetUp() {

}

void GpuModelHandlerLRIntegrationTest::TearDown() {

}

TEST_F(GpuModelHandlerLRIntegrationTest, 2Env_2SNP_no_recode) {

}

}
/* namespace Model */
} /* namespace CuEira */
