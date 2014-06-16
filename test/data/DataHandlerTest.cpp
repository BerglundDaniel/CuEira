#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>
#include <ostream>

#include <DataHandler.h>
#include <SNPVector.h>
#include <InteractionVector.h>
#include <EnvironmentVector.h>
#include <Id.h>
#include <SNP.h>
#include <EnvironmentFactor.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <Recode.h>
#include <InvalidState.h>
#include <StatisticModel.h>
#include <GeneticModel.h>
#include <RiskAllele.h>
#include <SNP.h>
#include <BedReaderMock.h>
#include <EnvironmentFactor.h>
#include <EnvironmentFactorHandlerMock.h>
#include <DataQueue.h>

using testing::Return;
using testing::AtLeast;
using testing::ReturnRef;

namespace CuEira {
namespace CuEira_Test {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataHandlerTest: public ::testing::Test {
protected:
  DataHandlerTest();
  virtual ~DataHandlerTest();
  virtual void SetUp();
  virtual void TearDown();

  const int numberOfSNPs;
  const int numberOfEnvironmentFactors;
  CuEira_Test::ConstructorHelpers constructorHelpers;
  EnvironmentFactorHandlerMock* environmentFactorHandlerMock;
  FileIO::BedReaderMock* bedReaderMock;
  std::vector<SNP*>* snpQueue;
  Task::DataQueue* dataQueue;
  std::vector<EnvironmentFactor*>* environmentInformation;
};

DataHandlerTest::DataHandlerTest() :
    numberOfSNPs(3), numberOfEnvironmentFactors(2), environmentFactorHandlerMock(
        constructorHelpers.constructEnvironmentFactorHandlerMock()), bedReaderMock(
        constructorHelpers.constructBedReaderMock()), snpQueue(nullptr), dataQueue(nullptr), environmentInformation(
        new std::vector<EnvironmentFactor*>(numberOfEnvironmentFactors)) {

  for(int i = 0; i < numberOfEnvironmentFactors; ++i){
    std::ostringstream os;
    os << "env" << i;
    Id id(os.str());
    (*environmentInformation)[i] = new EnvironmentFactor(id);
  }

}

DataHandlerTest::~DataHandlerTest() {
  delete environmentInformation;
}

void DataHandlerTest::SetUp() {
  snpQueue = new std::vector<SNP*>(numberOfSNPs);
  for(int i = 0; i < numberOfSNPs; ++i){
    std::ostringstream os;
    os << "snp" << i;
    Id id(os.str());

    (*snpQueue)[i] = new SNP(id, "allele1", "allele2", 1);
  }

  dataQueue = new Task::DataQueue(snpQueue);

  EXPECT_CALL(*environmentFactorHandlerMock, getHeaders()).Times(1).WillRepeatedly(ReturnRef(*environmentInformation));
}

void DataHandlerTest::TearDown() {
  delete dataQueue;
}

TEST_F(DataHandlerTest, ConstructAndGetException) {
  DataHandler dataHandler(ADDITIVE, bedReaderMock, environmentFactorHandlerMock, dataQueue);

  EXPECT_THROW(dataHandler.getCurrentEnvironmentFactor(), InvalidState);
  EXPECT_THROW(dataHandler.getCurrentSNP(), InvalidState);
  EXPECT_THROW(dataHandler.getEnvironment(), InvalidState);
  EXPECT_THROW(dataHandler.getSNP(), InvalidState);
  EXPECT_THROW(dataHandler.getInteraction(), InvalidState);
  EXPECT_THROW(dataHandler.getRecode(), InvalidState);
  EXPECT_THROW(dataHandler.recode(), InvalidState);
}

TEST_F(DataHandlerTest, Next) {
  DataHandler dataHandler(ADDITIVE, bedReaderMock, environmentFactorHandlerMock, dataQueue);

  //dataHandler.next();
}

TEST_F(DataHandlerTest, Recode) {
  DataHandler dataHandler(ADDITIVE, bedReaderMock, environmentFactorHandlerMock, dataQueue);
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */
