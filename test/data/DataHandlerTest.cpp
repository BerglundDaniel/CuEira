#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>
#include <ostream>

#include <DataHandler.h>
#include <SNPVector.h>
#include <SNPVectorMock.h>
#include <InteractionVector.h>
#include <InteractionVectorMock.h>
#include <EnvironmentVector.h>
#include <EnvironmentVectorMock.h>
#include <Id.h>
#include <SNP.h>
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
#include <DataQueue.h>
#include <ConstructorHelpers.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

using testing::Return;
using testing::AtLeast;
using testing::ReturnRef;
using testing::_;
using testing::Eq;

namespace CuEira {

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
  const int numberOfIndividuals;
  CuEira_Test::ConstructorHelpers constructorHelpers;
  FileIO::BedReaderMock* bedReaderMock;
  Container::EnvironmentVectorMock* environmentVectorMock;
  Container::InteractionVectorMock* interactionVectorMock;
  std::vector<SNP*>* snpQueue;
  std::vector<SNP*>* snpStore;
  Task::DataQueue* dataQueue;
  std::vector<const EnvironmentFactor*>* environmentInformation;
  std::vector<EnvironmentFactor*>* environmentStore;
};

DataHandlerTest::DataHandlerTest() :
    numberOfSNPs(3), numberOfEnvironmentFactors(2), numberOfIndividuals(5), constructorHelpers(), bedReaderMock(
        constructorHelpers.constructBedReaderMock()), snpQueue(nullptr), dataQueue(nullptr), environmentInformation(
        new std::vector<const EnvironmentFactor*>(numberOfEnvironmentFactors)), environmentStore(
        new std::vector<EnvironmentFactor*>(numberOfEnvironmentFactors)), snpStore(new std::vector<SNP*>(numberOfSNPs)), environmentVectorMock(
        nullptr), interactionVectorMock(nullptr) {

  for(int i = 0; i < numberOfEnvironmentFactors; ++i){
    std::ostringstream os;
    os << "env" << i;
    Id id(os.str());
    EnvironmentFactor* env = new EnvironmentFactor(id);
    (*environmentInformation)[i] = env;
    (*environmentStore)[i] = env;
  }

  for(int i = 0; i < numberOfSNPs; ++i){
    std::ostringstream os;
    os << "snp" << i;
    Id id(os.str());

    (*snpStore)[i] = new SNP(id, "allele1", "allele2", 1);
  }

}

DataHandlerTest::~DataHandlerTest() {
  for(int i = 0; i < numberOfEnvironmentFactors; ++i){
    delete (*environmentStore)[i];
  }

  delete environmentInformation;
  delete environmentStore;
  delete bedReaderMock;
}

void DataHandlerTest::SetUp() {
  environmentVectorMock = constructorHelpers.constructEnvironmentVectorMock();

  interactionVectorMock = new Container::InteractionVectorMock();

  snpQueue = new std::vector<SNP*>(numberOfSNPs);
  for(int i = 0; i < numberOfSNPs; ++i){
    (*snpQueue)[i] = (*snpStore)[i];
  }

  dataQueue = new Task::DataQueue(snpQueue);
}

void DataHandlerTest::TearDown() {
  for(int i = 0; i < numberOfSNPs; ++i){
    delete (*snpStore)[i];
  }

  delete snpStore;
  delete dataQueue;
}

#ifdef DEBUG
TEST_F(DataHandlerTest, ConstructAndGetException){
  DataHandler dataHandler(ADDITIVE, *bedReaderMock, *environmentInformation, *dataQueue, environmentVectorMock, interactionVectorMock);

  EXPECT_THROW(dataHandler.getCurrentEnvironmentFactor(), InvalidState);
  EXPECT_THROW(dataHandler.getCurrentSNP(), InvalidState);
  EXPECT_THROW(dataHandler.getEnvironment(), InvalidState);
  EXPECT_THROW(dataHandler.getSNP(), InvalidState);
  EXPECT_THROW(dataHandler.getInteraction(), InvalidState);
  EXPECT_THROW(dataHandler.getRecode(), InvalidState);
  EXPECT_THROW(dataHandler.recode(ALL_RISK), InvalidState);
  EXPECT_THROW(dataHandler.recode(SNP_PROTECT), InvalidState);
  EXPECT_THROW(dataHandler.recode(ENVIRONMENT_PROTECT), InvalidState);
  EXPECT_THROW(dataHandler.recode(INTERACTION_PROTECT), InvalidState);
}
#endif

TEST_F(DataHandlerTest, Next) {
  StatisticModel statisticModel = ADDITIVE;
  DataHandler dataHandler(statisticModel, *bedReaderMock, *environmentInformation, *dataQueue, environmentVectorMock,
      interactionVectorMock);

#ifdef CPU
  Container::HostVector* envData = new Container::LapackppHostVector(new LaVectorDouble(numberOfIndividuals));
  Container::HostVector* snpData = new Container::LapackppHostVector(new LaVectorDouble(numberOfIndividuals));
  Container::HostVector* interactionData = new Container::LapackppHostVector(new LaVectorDouble(numberOfIndividuals));
#else
  Container::HostVector* envData = new Container::PinnedHostVector(numberOfIndividuals);
  Container::HostVector* snpData = new Container::PinnedHostVector(numberOfIndividuals);
  Container::HostVector* interactionData = new Container::PinnedHostVector(numberOfIndividuals);
#endif

  Container::SNPVectorMock* snpVectorMock1 = constructorHelpers.constructSNPVectorMock();
  Container::SNPVectorMock* snpVectorMock2 = constructorHelpers.constructSNPVectorMock();

  EXPECT_CALL(*interactionVectorMock, getRecodedData()).Times(1 + 2 * (numberOfEnvironmentFactors + 1)).WillRepeatedly(
      ReturnRef(*interactionData));
  EXPECT_CALL(*interactionVectorMock, recode(_)).Times(numberOfEnvironmentFactors + 1);

  EXPECT_CALL(*environmentVectorMock, applyStatisticModel(statisticModel, _)).Times(numberOfEnvironmentFactors + 1);
  EXPECT_CALL(*environmentVectorMock, switchEnvironmentFactor(_)).Times(numberOfEnvironmentFactors + 1);

  EXPECT_CALL(*snpVectorMock1, applyStatisticModel(statisticModel, _)).Times(numberOfEnvironmentFactors);
  EXPECT_CALL(*snpVectorMock2, applyStatisticModel(statisticModel, _)).Times(1);

  EXPECT_CALL(*bedReaderMock, readSNP(Eq(*(*snpStore)[numberOfSNPs - 1]))).Times(1).WillRepeatedly(
      Return(snpVectorMock1));
  EXPECT_CALL(*bedReaderMock, readSNP(Eq(*(*snpStore)[numberOfSNPs - 2]))).Times(1).WillRepeatedly(
      Return(snpVectorMock2));

  ASSERT_TRUE(dataHandler.next());

  EXPECT_EQ(*(*environmentInformation)[0], dataHandler.getCurrentEnvironmentFactor());

  EXPECT_CALL(*snpVectorMock1, getAssociatedSNP()).Times(1).WillRepeatedly(ReturnRef(*(*snpStore)[numberOfSNPs - 1]));
  EXPECT_EQ(*(*snpStore)[numberOfSNPs - 1], dataHandler.getCurrentSNP());

  EXPECT_EQ(ALL_RISK, dataHandler.getRecode());

  EXPECT_CALL(*environmentVectorMock, getRecodedData()).Times(1).WillRepeatedly(ReturnRef(*envData));
  const Container::HostVector& envVector1 = dataHandler.getEnvironment();

  EXPECT_CALL(*snpVectorMock1, getRecodedData()).Times(1).WillRepeatedly(ReturnRef(*snpData));
  const Container::HostVector& snpVector1 = dataHandler.getSNP();

  const Container::HostVector& interactionVector1 = dataHandler.getInteraction();

  for(int i = 1; i < numberOfEnvironmentFactors; ++i){
    ASSERT_TRUE(dataHandler.next());
    EXPECT_EQ(*(*environmentInformation)[i], dataHandler.getCurrentEnvironmentFactor());
  }

  //Next snp
  dataHandler.currentRecode = SNP_PROTECT;
  ASSERT_TRUE(dataHandler.next());

  EXPECT_EQ(ALL_RISK, dataHandler.getRecode());
  EXPECT_EQ(*(*environmentInformation)[0], dataHandler.getCurrentEnvironmentFactor());

  EXPECT_CALL(*snpVectorMock2, getAssociatedSNP()).Times(1).WillRepeatedly(ReturnRef(*(*snpStore)[numberOfSNPs - 2]));
  EXPECT_EQ(*(*snpStore)[numberOfSNPs - 2], dataHandler.getCurrentSNP());

  delete envData;
  delete snpData;
  delete interactionData;
}

//TODO next snp include false

TEST_F(DataHandlerTest, Recode) {
#ifdef CPU
  Container::HostVector* interactionData = new Container::LapackppHostVector(new LaVectorDouble(numberOfIndividuals));
#else
  Container::HostVector* interactionData = new Container::PinnedHostVector(numberOfIndividuals);
#endif

  StatisticModel statisticModel = ADDITIVE;
  DataHandler dataHandler(statisticModel, *bedReaderMock, *environmentInformation, *dataQueue, environmentVectorMock,
      interactionVectorMock);
  dataHandler.state = dataHandler.INITIALISED;

  Container::SNPVectorMock* snpVectorMock = constructorHelpers.constructSNPVectorMock();
  dataHandler.snpVector = snpVectorMock;

  const int numberOfRecode = 4;
  Recode recodes[] = {SNP_PROTECT, ENVIRONMENT_PROTECT, INTERACTION_PROTECT, ALL_RISK};

  //Nothing should change
  dataHandler.recode(ALL_RISK);
  EXPECT_EQ(ALL_RISK, dataHandler.getRecode());

  EXPECT_CALL(*snpVectorMock, applyStatisticModel(statisticModel, _)).Times(numberOfRecode);
  EXPECT_CALL(*environmentVectorMock, applyStatisticModel(statisticModel, _)).Times(numberOfRecode);
  EXPECT_CALL(*interactionVectorMock, recode(_)).Times(numberOfRecode);
  EXPECT_CALL(*interactionVectorMock, getRecodedData()).Times(numberOfRecode * 2).WillRepeatedly(
      ReturnRef(*interactionData));

  for(int i = 0; i < numberOfRecode; ++i){
    Recode recode = recodes[i];

    EXPECT_CALL(*snpVectorMock, recode(Eq(recode))).Times(1);
    EXPECT_CALL(*environmentVectorMock, recode(Eq(recode))).Times(1);

    dataHandler.recode(recode);
    EXPECT_EQ(recode, dataHandler.getRecode());
  }

  delete interactionData;
}

TEST_F(DataHandlerTest, InteractionTestNoMocks) {
  //DataHandler dataHandler(ADDITIVE, *bedReaderMock, *environmentInformation, *dataQueue, environmentVectorMock, interactionVectorMock);
}

} /* namespace CuEira */
