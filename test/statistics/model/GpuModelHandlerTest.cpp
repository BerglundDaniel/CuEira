#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <ostream>

#include <HostVector.h>
#include <HostMatrix.h>
#include <DataHandlerMock.h>
#include <LogisticRegressionMock.h>
#include <LogisticRegressionConfigurationMock.h>
#include <LogisticRegressionResultMock.h>
#include <SNP.h>
#include <Id.h>
#include <EnvironmentFactor.h>
#include <GpuModelHandler.h>
#include <ModelHandler.h>
#include <Statistics.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

using testing::Return;
using testing::ReturnRef;
using testing::_;
using ::testing::Ge;
using ::testing::Eq;
using ::testing::Le;

namespace CuEira {
namespace Model {

MATCHER_P(CompareAddress, value, "Compares the address of the args"){
return &value == &arg;
}

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

  DataHandlerMock* dataHandlerMock;
  LogisticRegression::LogisticRegressionMock* logisticRegressionMock;
  LogisticRegression::LogisticRegressionConfigurationMock* logisticRegressionConfigurationMock;
  LogisticRegression::LogisticRegressionResultMock* logisticRegressionResultMock;
  const int numberOfRows;
  const int numberOfPredictors;
};

GpuModelHandlerTest::GpuModelHandlerTest() :
    logisticRegressionMock(nullptr), logisticRegressionConfigurationMock(nullptr), logisticRegressionResultMock(
        nullptr), dataHandlerMock(nullptr), numberOfRows(5), numberOfPredictors(4) {

}

GpuModelHandlerTest::~GpuModelHandlerTest() {

}

void GpuModelHandlerTest::SetUp() {
  dataHandlerMock = new DataHandlerMock();
  logisticRegressionMock = new LogisticRegression::LogisticRegressionMock(); //FIXME
  logisticRegressionConfigurationMock = new LogisticRegression::LogisticRegressionConfigurationMock(); //FIXME
  logisticRegressionResultMock = new LogisticRegression::LogisticRegressionResultMock();

  EXPECT_CALL(*logisticRegressionConfigurationMock, getNumberOfRows()).Times(1).WillRepeatedly(Return(numberOfRows));
  EXPECT_CALL(*logisticRegressionConfigurationMock, getNumberOfPredictors()).Times(1).WillRepeatedly(
      Return(numberOfPredictors));
}

void GpuModelHandlerTest::TearDown() {

}

TEST_F(GpuModelHandlerTest, Next) {
  GpuModelHandler gpuModelHandler(dataHandlerMock, *logisticRegressionConfigurationMock, logisticRegressionMock);
  ASSERT_EQ(gpuModelHandler.NOT_INITIALISED, gpuModelHandler.state);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  EnvironmentFactor envFactor(Id("env1"));
  Container::PinnedHostVector snpData(numberOfRows);
  Container::PinnedHostVector envData(numberOfRows);
  Container::PinnedHostVector interactionData(numberOfRows);

  EXPECT_CALL(*dataHandlerMock, next()).Times(1).WillRepeatedly(Return(true));

  EXPECT_CALL(*dataHandlerMock, getSNP()).Times(1).WillRepeatedly(ReturnRef(snpData));
  EXPECT_CALL(*dataHandlerMock, getEnvironment()).Times(1).WillRepeatedly(ReturnRef(envData));
  EXPECT_CALL(*dataHandlerMock, getInteraction()).Times(1).WillRepeatedly(ReturnRef(interactionData));

  EXPECT_CALL(*dataHandlerMock, getCurrentSNP()).Times(1).WillRepeatedly(ReturnRef(snp));
  EXPECT_CALL(*dataHandlerMock, getCurrentEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(envFactor));

  ASSERT_TRUE(gpuModelHandler.next());
  ASSERT_EQ(gpuModelHandler.INITIALISED_READY, gpuModelHandler.state);
}

TEST_F(GpuModelHandlerTest, NextFalse) {

  GpuModelHandler gpuModelHandler(dataHandlerMock, *logisticRegressionConfigurationMock, logisticRegressionMock);

  EXPECT_CALL(*dataHandlerMock, next()).Times(1).WillRepeatedly(Return(false));

  ASSERT_FALSE(gpuModelHandler.next());
}

TEST_F(GpuModelHandlerTest, NextAndCalculate) {
  GpuModelHandler gpuModelHandler(dataHandlerMock, *logisticRegressionConfigurationMock, logisticRegressionMock);
  ASSERT_EQ(gpuModelHandler.NOT_INITIALISED, gpuModelHandler.state);

  SNP snp1(Id("snp1"), "a1", "a2", 1);
  EnvironmentFactor envFactor1(Id("env1"));
  Container::PinnedHostVector snpData1(numberOfRows);
  Container::PinnedHostVector envData1(numberOfRows);
  Container::PinnedHostVector interactionData1(numberOfRows);

  SNP snp2(Id("snp2"), "a1", "a2", 1);
  EnvironmentFactor envFactor2(Id("env2"));
  Container::PinnedHostVector snpData2(numberOfRows);
  Container::PinnedHostVector envData2(numberOfRows);
  Container::PinnedHostVector interactionData2(numberOfRows);

  EXPECT_CALL(*dataHandlerMock, next()).Times(2).WillRepeatedly(Return(true));

  EXPECT_CALL(*dataHandlerMock, getSNP()).Times(1).WillRepeatedly(ReturnRef(snpData1));
  EXPECT_CALL(*dataHandlerMock, getEnvironment()).Times(1).WillRepeatedly(ReturnRef(envData1));
  EXPECT_CALL(*dataHandlerMock, getInteraction()).Times(1).WillRepeatedly(ReturnRef(interactionData1));

  EXPECT_CALL(*dataHandlerMock, getCurrentSNP()).Times(1).WillRepeatedly(ReturnRef(snp1));
  EXPECT_CALL(*dataHandlerMock, getCurrentEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(envFactor1));

  ASSERT_TRUE(gpuModelHandler.next());
  ASSERT_EQ(gpuModelHandler.INITIALISED_READY, gpuModelHandler.state);

  EXPECT_CALL(*logisticRegressionConfigurationMock, setSNP(CompareAddress(snpData1))).Times(1);
  EXPECT_CALL(*logisticRegressionConfigurationMock, setEnvironmentFactor(CompareAddress(envData1))).Times(1);
  EXPECT_CALL(*logisticRegressionConfigurationMock, setInteraction(CompareAddress(interactionData1))).Times(1);

  EXPECT_CALL(*logisticRegressionMock, calculate()).Times(2).WillRepeatedly(Return(logisticRegressionResultMock));
  EXPECT_CALL(*logisticRegressionResultMock, calculateRecode()).Times(2).WillRepeatedly(Return(ALL_RISK));

  Statistics* statistics1 = gpuModelHandler.calculateModel();

  EXPECT_CALL(*dataHandlerMock, getSNP()).Times(1).WillRepeatedly(ReturnRef(snpData2));
  EXPECT_CALL(*dataHandlerMock, getEnvironment()).Times(1).WillRepeatedly(ReturnRef(envData2));
  EXPECT_CALL(*dataHandlerMock, getInteraction()).Times(1).WillRepeatedly(ReturnRef(interactionData2));

  EXPECT_CALL(*dataHandlerMock, getCurrentSNP()).Times(1).WillRepeatedly(ReturnRef(snp2));
  EXPECT_CALL(*dataHandlerMock, getCurrentEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(envFactor2));

  ASSERT_TRUE(gpuModelHandler.next());
  ASSERT_EQ(gpuModelHandler.INITIALISED_FULL, gpuModelHandler.state);

  EXPECT_CALL(*logisticRegressionConfigurationMock, setSNP(CompareAddress(snpData2))).Times(1);
  EXPECT_CALL(*logisticRegressionConfigurationMock, setEnvironmentFactor(CompareAddress(envData2))).Times(1);
  EXPECT_CALL(*logisticRegressionConfigurationMock, setInteraction(CompareAddress(interactionData2))).Times(1);

  Statistics* statistics2 = gpuModelHandler.calculateModel();

  delete statistics1;
  delete statistics2;
}

TEST_F(GpuModelHandlerTest, NextAndCalculateRecode) {
  GpuModelHandler gpuModelHandler(dataHandlerMock, *logisticRegressionConfigurationMock, logisticRegressionMock);

  SNP snp(Id("snp1"), "a1", "a2", 1);
  EnvironmentFactor envFactor(Id("env1"));
  Container::PinnedHostVector snpData(numberOfRows);
  Container::PinnedHostVector envData(numberOfRows);
  Container::PinnedHostVector interactionData(numberOfRows);

  EXPECT_CALL(*dataHandlerMock, next()).Times(1).WillRepeatedly(Return(true));

  EXPECT_CALL(*dataHandlerMock, getSNP()).Times(1).WillRepeatedly(ReturnRef(snpData));
  EXPECT_CALL(*dataHandlerMock, getEnvironment()).Times(1).WillRepeatedly(ReturnRef(envData));
  EXPECT_CALL(*dataHandlerMock, getInteraction()).Times(1).WillRepeatedly(ReturnRef(interactionData));

  EXPECT_CALL(*dataHandlerMock, getCurrentSNP()).Times(1).WillRepeatedly(ReturnRef(snp));
  EXPECT_CALL(*dataHandlerMock, getCurrentEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(envFactor));

  ASSERT_TRUE(gpuModelHandler.next());

  EXPECT_CALL(*logisticRegressionConfigurationMock, setSNP(CompareAddress(snpData))).Times(1);
  EXPECT_CALL(*logisticRegressionConfigurationMock, setEnvironmentFactor(CompareAddress(envData))).Times(1);
  EXPECT_CALL(*logisticRegressionConfigurationMock, setInteraction(CompareAddress(interactionData))).Times(1);

  EXPECT_CALL(*logisticRegressionMock, calculate()).Times(2).WillRepeatedly(Return(logisticRegressionResultMock));
  EXPECT_CALL(*logisticRegressionResultMock, calculateRecode()).Times(1).WillRepeatedly(Return(INTERACTION_PROTECT));

  EXPECT_CALL(*dataHandlerMock, recode(INTERACTION_PROTECT)).Times(1);

  Statistics* statistics = gpuModelHandler.calculateModel();

  delete statistics;
}

}
/* namespace Model */
} /* namespace CuEira */
