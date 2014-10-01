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
#include <StatisticsFactoryMock.h>

#include <PinnedHostVector.h>

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
  ModelStatisticsFactoryMock* statisticsFactoryMock;
  const int numberOfRows;
  const int numberOfPredictors;
};

GpuModelHandlerTest::GpuModelHandlerTest() :
    logisticRegressionMock(nullptr), logisticRegressionConfigurationMock(nullptr), logisticRegressionResultMock(
        nullptr), dataHandlerMock(nullptr), numberOfRows(5), numberOfPredictors(4), statisticsFactoryMock(nullptr) {

}

GpuModelHandlerTest::~GpuModelHandlerTest() {

}

void GpuModelHandlerTest::SetUp() {
  dataHandlerMock = new DataHandlerMock();
  logisticRegressionMock = new LogisticRegression::LogisticRegressionMock();
  logisticRegressionConfigurationMock = new LogisticRegression::LogisticRegressionConfigurationMock();
  logisticRegressionResultMock = new LogisticRegression::LogisticRegressionResultMock();
  statisticsFactoryMock = new ModelStatisticsFactoryMock();

  EXPECT_CALL(*logisticRegressionConfigurationMock, getNumberOfRows()).Times(1).WillRepeatedly(Return(numberOfRows));
  EXPECT_CALL(*logisticRegressionConfigurationMock, getNumberOfPredictors()).Times(1).WillRepeatedly(
      Return(numberOfPredictors));
}

void GpuModelHandlerTest::TearDown() {
  delete logisticRegressionConfigurationMock;
  delete statisticsFactoryMock;
  delete logisticRegressionResultMock;
}

TEST_F(GpuModelHandlerTest, Next) {
  GpuModelHandler gpuModelHandler(*statisticsFactoryMock, dataHandlerMock, *logisticRegressionConfigurationMock,
      logisticRegressionMock);
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

  GpuModelHandler gpuModelHandler(*statisticsFactoryMock, dataHandlerMock, *logisticRegressionConfigurationMock,
      logisticRegressionMock);

  EXPECT_CALL(*dataHandlerMock, next()).Times(1).WillRepeatedly(Return(false));

  ASSERT_FALSE(gpuModelHandler.next());
}

TEST_F(GpuModelHandlerTest, NextAndCalculate) {
  GpuModelHandler gpuModelHandler(*statisticsFactoryMock, dataHandlerMock, *logisticRegressionConfigurationMock,
      logisticRegressionMock);
  ASSERT_EQ(gpuModelHandler.NOT_INITIALISED, gpuModelHandler.state);

  SNP snp1(Id("snp1"), "a1", "a2", 1);
  EnvironmentFactor envFactor1(Id("env1"));
  Container::PinnedHostVector snpData1(numberOfRows);
  Container::PinnedHostVector envData1(numberOfRows + 1);
  Container::PinnedHostVector interactionData1(numberOfRows + 2);

  SNP snp2(Id("snp2"), "a1", "a2", 1);
  EnvironmentFactor envFactor2(Id("env2"));
  Container::PinnedHostVector snpData2(numberOfRows + 3);
  Container::PinnedHostVector envData2(numberOfRows + 4);
  Container::PinnedHostVector interactionData2(numberOfRows + 5);

  EXPECT_CALL(*dataHandlerMock, next()).Times(2).WillRepeatedly(Return(true));
  EXPECT_CALL(*dataHandlerMock, getSNP()).Times(1).WillRepeatedly(ReturnRef(snpData1));
  EXPECT_CALL(*dataHandlerMock, getEnvironment()).Times(1).WillRepeatedly(ReturnRef(envData1));
  EXPECT_CALL(*dataHandlerMock, getInteraction()).Times(1).WillRepeatedly(ReturnRef(interactionData1));

  EXPECT_CALL(*dataHandlerMock, getCurrentSNP()).Times(1).WillRepeatedly(ReturnRef(snp1));
  EXPECT_CALL(*dataHandlerMock, getCurrentEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(envFactor1));

  ASSERT_TRUE(gpuModelHandler.next());
  ASSERT_EQ(gpuModelHandler.INITIALISED_READY, gpuModelHandler.state);

  EXPECT_CALL(*logisticRegressionConfigurationMock, setSNP(_)).Times(2);
  EXPECT_CALL(*logisticRegressionConfigurationMock, setEnvironmentFactor(_)).Times(2);
  EXPECT_CALL(*logisticRegressionConfigurationMock, setInteraction(_)).Times(2);

  EXPECT_CALL(*logisticRegressionMock, calculate()).Times(2).WillRepeatedly(Return(logisticRegressionResultMock));
  EXPECT_CALL(*logisticRegressionResultMock, calculateRecode()).Times(2).WillRepeatedly(Return(ALL_RISK));
  EXPECT_CALL(*statisticsFactoryMock, constructStatistics(_)).Times(2).WillRepeatedly(Return(nullptr));

  gpuModelHandler.calculateModel();

  EXPECT_CALL(*dataHandlerMock, getSNP()).Times(1).WillRepeatedly(ReturnRef(snpData2));
  EXPECT_CALL(*dataHandlerMock, getEnvironment()).Times(1).WillRepeatedly(ReturnRef(envData2));
  EXPECT_CALL(*dataHandlerMock, getInteraction()).Times(1).WillRepeatedly(ReturnRef(interactionData2));

  EXPECT_CALL(*dataHandlerMock, getCurrentSNP()).Times(1).WillRepeatedly(ReturnRef(snp2));
  EXPECT_CALL(*dataHandlerMock, getCurrentEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(envFactor2));

  ASSERT_TRUE(gpuModelHandler.next());
  ASSERT_EQ(gpuModelHandler.INITIALISED_FULL, gpuModelHandler.state);

  gpuModelHandler.calculateModel();
}

TEST_F(GpuModelHandlerTest, NextAndCalculateRecode) {
  GpuModelHandler gpuModelHandler(*statisticsFactoryMock, dataHandlerMock, *logisticRegressionConfigurationMock,
      logisticRegressionMock);
  LogisticRegression::LogisticRegressionResultMock* logisticRegressionResultMock2 =
      new LogisticRegression::LogisticRegressionResultMock();

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

  EXPECT_CALL(*logisticRegressionConfigurationMock, setSNP(_)).Times(2);
  EXPECT_CALL(*logisticRegressionConfigurationMock, setEnvironmentFactor(_)).Times(2);
  EXPECT_CALL(*logisticRegressionConfigurationMock, setInteraction(_)).Times(2);

  {
    InSequence s;

    EXPECT_CALL(*logisticRegressionMock, calculate()).Times(1).WillOnce(Return(logisticRegressionResultMock2));
    EXPECT_CALL(*logisticRegressionMock, calculate()).Times(1).WillOnce(Return(logisticRegressionResultMock));
  }

  EXPECT_CALL(*logisticRegressionResultMock2, calculateRecode()).Times(1).WillRepeatedly(Return(INTERACTION_PROTECT));
  EXPECT_CALL(*statisticsFactoryMock, constructStatistics(_)).Times(1).WillRepeatedly(Return(nullptr));

  EXPECT_CALL(*dataHandlerMock, recode(INTERACTION_PROTECT)).Times(1);

  gpuModelHandler.calculateModel();
}

}
/* namespace Model */
} /* namespace CuEira */
