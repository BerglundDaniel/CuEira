#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utility>
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
#include <ConfigurationMock.h>
#include <ContingencyTableMock.h>
#include <ContingencyTableFactoryMock.h>
#include <ModelState.h>
#include <AlleleStatisticsMock.h>
#include <ModelInformation.h>
#include <ModelInformationMock.h>
#include <ModelInformationFactoryMock.h>

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
using testing::Sequence;
using testing::SaveArg;
using testing::DoAll;
using testing::ByRef;

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
  ConfigurationMock* configurationMock;
  ContingencyTableFactoryMock* contingencyTableFactoryMock;
  Model::ModelInformationFactoryMock* modelInformationFactoryMock;
  std::vector<SNP*>* snpQueue;
  std::vector<SNP*>* snpStore;
  Task::DataQueue* dataQueue;
  std::vector<const EnvironmentFactor*>* environmentInformation;
  std::vector<EnvironmentFactor*>* environmentStore;
};

DataHandlerTest::DataHandlerTest() :
    numberOfSNPs(2), numberOfEnvironmentFactors(3), numberOfIndividuals(5), constructorHelpers(), bedReaderMock(
        constructorHelpers.constructBedReaderMock()), snpQueue(nullptr), dataQueue(nullptr), environmentInformation(
        new std::vector<const EnvironmentFactor*>(numberOfEnvironmentFactors)), environmentStore(
        new std::vector<EnvironmentFactor*>(numberOfEnvironmentFactors)), snpStore(nullptr), environmentVectorMock(
        nullptr), interactionVectorMock(nullptr), configurationMock(new ConfigurationMock()), contingencyTableFactoryMock(
        constructorHelpers.constructContingencyTableFactoryMock()), modelInformationFactoryMock(
        new Model::ModelInformationFactoryMock) {

  for(int i = 0; i < numberOfEnvironmentFactors; ++i){
    std::ostringstream os;
    os << "env" << i;
    Id id(os.str());
    EnvironmentFactor* env = new EnvironmentFactor(id);
    (*environmentInformation)[i] = env;
    (*environmentStore)[i] = env;
  }

}

DataHandlerTest::~DataHandlerTest() {
  for(int i = 0; i < numberOfEnvironmentFactors; ++i){
    delete (*environmentStore)[i];
  }

  delete environmentInformation;
  delete environmentStore;
  delete bedReaderMock;
  delete configurationMock;
  delete contingencyTableFactoryMock;
  delete modelInformationFactoryMock;
}

void DataHandlerTest::SetUp() {
  environmentVectorMock = constructorHelpers.constructEnvironmentVectorMock();

  interactionVectorMock = new Container::InteractionVectorMock();

  snpQueue = new std::vector<SNP*>(numberOfSNPs);
  snpStore = new std::vector<SNP*>(numberOfSNPs);

  for(int i = 0; i < numberOfSNPs; ++i){
    std::ostringstream os;
    os << "snp" << i;
    Id id(os.str());
    SNP* snp = new SNP(id, "allele1", "allele2", 1);

    (*snpStore)[numberOfSNPs - i - 1] = snp;
    (*snpQueue)[i] = snp;
  }

  dataQueue = new Task::DataQueue(snpQueue);
}

void DataHandlerTest::TearDown() {
  SNP* snp = dataQueue->next();
  while(snp != nullptr){
    delete snp;
    snp = dataQueue->next();
  }

  delete snpStore;
  delete dataQueue;
}

#ifdef DEBUG
TEST_F(DataHandlerTest, ConstructAndGetException){
  EXPECT_CALL(*configurationMock, getStatisticModel()).Times(1).WillRepeatedly(Return(ADDITIVE));
  EXPECT_CALL(*configurationMock, getCellCountThreshold()).Times(1).WillRepeatedly(Return(0));

  DataHandler dataHandler(*configurationMock, *bedReaderMock, *contingencyTableFactoryMock, *modelInformationFactoryMock, *environmentInformation, *dataQueue, environmentVectorMock, interactionVectorMock);

  EXPECT_THROW(dataHandler.getCurrentEnvironmentFactor(), InvalidState);
  EXPECT_THROW(dataHandler.getCurrentSNP(), InvalidState);
  EXPECT_THROW(dataHandler.getEnvironmentVector(), InvalidState);
  EXPECT_THROW(dataHandler.getSNPVector(), InvalidState);
  EXPECT_THROW(dataHandler.getInteractionVector(), InvalidState);
  EXPECT_THROW(dataHandler.getRecode(), InvalidState);
  EXPECT_THROW(dataHandler.getContingencyTable(), InvalidState);
  EXPECT_THROW(dataHandler.getAlleleStatistics(), InvalidState);
  EXPECT_THROW(dataHandler.recode(ALL_RISK), InvalidState);
  EXPECT_THROW(dataHandler.recode(SNP_PROTECT), InvalidState);
  EXPECT_THROW(dataHandler.recode(ENVIRONMENT_PROTECT), InvalidState);
  EXPECT_THROW(dataHandler.recode(INTERACTION_PROTECT), InvalidState);
}
#endif

TEST_F(DataHandlerTest, Next) {
  const StatisticModel statisticModel = ADDITIVE;
  EXPECT_CALL(*configurationMock, getStatisticModel()).Times(1).WillRepeatedly(Return(statisticModel));
  EXPECT_CALL(*configurationMock, getCellCountThreshold()).Times(1).WillRepeatedly(Return(0));
  Sequence readSequence;
  Sequence contingencyTableSequence;
  Sequence modelInformationSequence;
  const int numberOfRuns = numberOfEnvironmentFactors * numberOfSNPs;
  std::vector<ContingencyTableMock*> contingencyTableVector(numberOfRuns);
  std::vector<Model::ModelInformationMock*> modelInformationVector(numberOfRuns);

  DataHandler dataHandler(*configurationMock, *bedReaderMock, *contingencyTableFactoryMock,
      *modelInformationFactoryMock, *environmentInformation, *dataQueue, environmentVectorMock, interactionVectorMock);

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

  AlleleStatisticsMock* alleleStatisticsMock1 = new AlleleStatisticsMock();
  AlleleStatisticsMock* alleleStatisticsMock2 = new AlleleStatisticsMock();

  for(int i = 0; i < numberOfRuns; ++i){
    contingencyTableVector[i] = new ContingencyTableMock();
    modelInformationVector[i] = new Model::ModelInformationMock();
  }

  std::vector<int> contingencyTable_Table(8);
  for(int i = 0; i < 8; ++i){
    contingencyTable_Table[i] = 1;
  }

  std::pair<const AlleleStatistics*, Container::SNPVector*>* pair1 = new std::pair<const AlleleStatistics*,
      Container::SNPVector*>();
  std::pair<const AlleleStatistics*, Container::SNPVector*>* pair2 = new std::pair<const AlleleStatistics*,
      Container::SNPVector*>();

  pair1->first = alleleStatisticsMock1;
  pair1->second = snpVectorMock1;
  pair2->first = alleleStatisticsMock2;
  pair2->second = snpVectorMock2;

  EXPECT_CALL(*interactionVectorMock, getRecodedData()).Times(2 * (numberOfRuns)).WillRepeatedly(
      ReturnRef(*interactionData));
  EXPECT_CALL(*interactionVectorMock, recode(_)).Times(numberOfRuns);

  EXPECT_CALL(*environmentVectorMock, applyStatisticModel(statisticModel, _)).Times(numberOfRuns);
  EXPECT_CALL(*environmentVectorMock, switchEnvironmentFactor(_)).Times(numberOfRuns);

  EXPECT_CALL(*snpVectorMock1, applyStatisticModel(statisticModel, _)).Times(numberOfEnvironmentFactors);
  EXPECT_CALL(*snpVectorMock1, recode(ALL_RISK)).Times(numberOfEnvironmentFactors - 1);
  EXPECT_CALL(*snpVectorMock2, applyStatisticModel(statisticModel, _)).Times(numberOfEnvironmentFactors);
  EXPECT_CALL(*snpVectorMock2, recode(ALL_RISK)).Times(numberOfEnvironmentFactors - 1);

  for(int i = 0; i < numberOfRuns; ++i){
    ContingencyTableMock * contingencyTableMock = contingencyTableVector[i];

    EXPECT_CALL(*contingencyTableFactoryMock, constructContingencyTable(_, _)).Times(1).InSequence(
        contingencyTableSequence).WillOnce(Return(contingencyTableMock));
    EXPECT_CALL(*contingencyTableMock, getTable()).Times(1).InSequence(contingencyTableSequence).WillOnce(
        ReturnRef(contingencyTable_Table));
    EXPECT_CALL(*contingencyTableMock, Die()).InSequence(contingencyTableSequence);

    Model::ModelInformationMock* modelInformationMock = modelInformationVector[i];

    if(i < numberOfEnvironmentFactors){
      EXPECT_CALL(*modelInformationFactoryMock, constructModelInformation(CALCULATE,Eq(ByRef(*(*snpStore)[0])),Eq(ByRef(*(*environmentStore)[i])),_,_)).Times(
          1).InSequence(modelInformationSequence).WillOnce(Return(modelInformationMock));
    }else{
      EXPECT_CALL(*modelInformationFactoryMock, constructModelInformation(CALCULATE,Eq(ByRef(*(*snpStore)[1])),Eq(ByRef(*(*environmentStore)[i-numberOfEnvironmentFactors])),_,_)).Times(
          1).InSequence(modelInformationSequence).WillOnce(Return(modelInformationMock));
    }
  }

  EXPECT_CALL(*bedReaderMock, readSNP(Eq(ByRef(*(*snpStore)[0])))).Times(1).InSequence(readSequence).WillOnce(
      Return(pair1));
  EXPECT_CALL(*bedReaderMock, readSNP(Eq(ByRef(*(*snpStore)[1])))).Times(1).InSequence(readSequence).WillOnce(
      Return(pair2));

  Model::ModelInformation* modelInformation = dataHandler.next();
  EXPECT_EQ(modelInformationVector[0], modelInformation);
  delete modelInformation;

  EXPECT_EQ(*(*environmentInformation)[0], dataHandler.getCurrentEnvironmentFactor());
  EXPECT_EQ(*(*snpStore)[0], dataHandler.getCurrentSNP());
  EXPECT_EQ(ALL_RISK, dataHandler.getRecode());

  for(int i = 1; i < numberOfEnvironmentFactors; ++i){
    modelInformation = dataHandler.next();
    EXPECT_EQ(*(*environmentInformation)[i], dataHandler.getCurrentEnvironmentFactor());
    EXPECT_EQ(modelInformationVector[i], modelInformation);

    delete modelInformation;
  }

  //Next snp
  for(int i = 0; i < numberOfEnvironmentFactors; ++i){
    modelInformation = dataHandler.next();
    EXPECT_EQ(ALL_RISK, dataHandler.getRecode());
    EXPECT_EQ(*(*environmentInformation)[i], dataHandler.getCurrentEnvironmentFactor());
    EXPECT_EQ(*(*snpStore)[1], dataHandler.getCurrentSNP());
    EXPECT_EQ(modelInformationVector[numberOfEnvironmentFactors + i], modelInformation);

    delete modelInformation;
  }

  //Next done
  Model::ModelInformation* modelInformationDoneMock = new Model::ModelInformationMock();
  EXPECT_CALL(*modelInformationFactoryMock, constructModelInformation(DONE)).Times(1).InSequence(
      modelInformationSequence).WillOnce(Return(modelInformationDoneMock));
  modelInformation = dataHandler.next();
  EXPECT_EQ(modelInformationDoneMock, modelInformation);
  delete modelInformation;

  delete envData;
  delete snpData;
  delete interactionData;
}

TEST_F(DataHandlerTest, Recode) {
#ifdef CPU
  Container::HostVector* interactionData = new Container::LapackppHostVector(new LaVectorDouble(numberOfIndividuals));
#else
  Container::HostVector* interactionData = new Container::PinnedHostVector(numberOfIndividuals);
#endif

  const StatisticModel statisticModel = ADDITIVE;
  EXPECT_CALL(*configurationMock, getStatisticModel()).Times(1).WillRepeatedly(Return(statisticModel));
  EXPECT_CALL(*configurationMock, getCellCountThreshold()).Times(1).WillRepeatedly(Return(0));

  DataHandler dataHandler(*configurationMock, *bedReaderMock, *contingencyTableFactoryMock,
      *modelInformationFactoryMock, *environmentInformation, *dataQueue, environmentVectorMock, interactionVectorMock);
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

TEST_F(DataHandlerTest, ReadSNPIncludeFalse) {
  const StatisticModel statisticModel = ADDITIVE;
  (*snpStore)[0]->setInclude(MISSING_DATA);
  EXPECT_CALL(*configurationMock, getStatisticModel()).Times(1).WillRepeatedly(Return(statisticModel));
  EXPECT_CALL(*configurationMock, getCellCountThreshold()).Times(1).WillRepeatedly(Return(0));

  DataHandler dataHandler(*configurationMock, *bedReaderMock, *contingencyTableFactoryMock,
      *modelInformationFactoryMock, *environmentInformation, *dataQueue, environmentVectorMock, interactionVectorMock);

  Container::SNPVectorMock* snpVectorMock1 = constructorHelpers.constructSNPVectorMock();
  AlleleStatisticsMock* alleleStatisticsMock1 = new AlleleStatisticsMock();
  std::pair<const AlleleStatistics*, Container::SNPVector*>* pair1 = new std::pair<const AlleleStatistics*,
      Container::SNPVector*>();

  pair1->first = alleleStatisticsMock1;
  pair1->second = snpVectorMock1;

  EXPECT_CALL(*bedReaderMock, readSNP(Eq(ByRef(*(*snpStore)[0])))).Times(1).WillOnce(Return(pair1));

  Model::ModelInformation* modelInformationSkipMock = new Model::ModelInformationMock();
  EXPECT_CALL(*modelInformationFactoryMock, constructModelInformation(SKIP,Eq(ByRef(*(*snpStore)[0])),Eq(ByRef(*(*environmentStore)[0])),_)).Times(
      1).WillOnce(Return(modelInformationSkipMock));
  Model::ModelInformation* modelInformation = dataHandler.next();
  EXPECT_EQ(modelInformationSkipMock, modelInformation);

  delete modelInformation;
}

TEST_F(DataHandlerTest, ContingencyTableIncludeFalse) {
  const StatisticModel statisticModel = ADDITIVE;
  EXPECT_CALL(*configurationMock, getStatisticModel()).Times(1).WillRepeatedly(Return(statisticModel));
  EXPECT_CALL(*configurationMock, getCellCountThreshold()).Times(1).WillRepeatedly(Return(5));

  DataHandler dataHandler(*configurationMock, *bedReaderMock, *contingencyTableFactoryMock,
      *modelInformationFactoryMock, *environmentInformation, *dataQueue, environmentVectorMock, interactionVectorMock);

  Container::SNPVectorMock* snpVectorMock1 = constructorHelpers.constructSNPVectorMock();
  AlleleStatisticsMock* alleleStatisticsMock1 = new AlleleStatisticsMock();
  std::pair<const AlleleStatistics*, Container::SNPVector*>* pair1 = new std::pair<const AlleleStatistics*,
      Container::SNPVector*>();

  pair1->first = alleleStatisticsMock1;
  pair1->second = snpVectorMock1;

  EXPECT_CALL(*bedReaderMock, readSNP(Eq(ByRef(*(*snpStore)[0])))).Times(1).WillOnce(Return(pair1));

  ContingencyTableMock* contingencyTable = new ContingencyTableMock();

  std::vector<int> contingencyTable_Table(8);
  for(int i = 0; i < 8; ++i){
    contingencyTable_Table[i] = 1;
  }

  EXPECT_CALL(*environmentVectorMock, switchEnvironmentFactor(_)).Times(1);
  EXPECT_CALL(*interactionVectorMock, recode(_)).Times(1);

  EXPECT_CALL(*contingencyTableFactoryMock, constructContingencyTable(_, _)).Times(1).WillOnce(
      Return(contingencyTable));
  EXPECT_CALL(*contingencyTable, getTable()).Times(1).WillOnce(ReturnRef(contingencyTable_Table));

  Model::ModelInformation* modelInformationSkipMock = new Model::ModelInformationMock();
  EXPECT_CALL(*modelInformationFactoryMock, constructModelInformation(SKIP,Eq(ByRef(*(*snpStore)[0])),Eq(ByRef(*(*environmentStore)[0])),_,_)).Times(
      1).WillOnce(Return(modelInformationSkipMock));
  Model::ModelInformation* modelInformation = dataHandler.next();
  EXPECT_EQ(modelInformationSkipMock, modelInformation);

  delete modelInformation;
}

} /* namespace CuEira */
