#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <thread>
#include <utility>

#include <BedReader.h>
#include <ConfigurationMock.h>
#include <PersonHandler.h>
#include <GeneticModel.h>
#include <Person.h>
#include <Sex.h>
#include <Id.h>
#include <Phenotype.h>
#include <SNP.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <SNPVector.h>
#include <SNPVectorFactory.h>
#include <AlleleStatisticsFactory.h>
#include <AlleleStatistics.h>
#include <EnvironmentFactorHandler.h>
#include <EnvironmentFactor.h>
#include <DataFilesReader.h>
#include <DataFilesReaderFactory.h>

using testing::Return;
using testing::_;
using testing::ReturnRef;
using testing::Eq;
using testing::SaveArg;
using testing::DoAll;
using testing::AtLeast;

namespace CuEira {

namespace CuEira_Test {

void threadBedReaderTest(const FileIO::BedReader* bedReader, SNP* snp, std::vector<Container::SNPVector*>* snpVectors,
    const int numberOfReads) {

  for(int i = 0; i < numberOfReads; ++i){
    std::pair<const AlleleStatistics*, Container::SNPVector*>* pair = bedReader->readSNP(*snp);

    delete pair->first; //FIXME need to check this too
    (*snpVectors)[i] = pair->second;

    delete pair;
  }
}

}

namespace FileIO {

/**
 * Test for testing BedReader while multiple threads are sharing the same BedReader object. Assumes the other fileReaders are working properly.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class BedReaderThreadIntegrationTest: public ::testing::Test {
protected:
  BedReaderThreadIntegrationTest();
  virtual ~BedReaderThreadIntegrationTest();
  virtual void SetUp();
  virtual void TearDown();

  ConfigurationMock configMock;
};

BedReaderThreadIntegrationTest::BedReaderThreadIntegrationTest() {

}

BedReaderThreadIntegrationTest::~BedReaderThreadIntegrationTest() {
}

void BedReaderThreadIntegrationTest::SetUp() {
//Expect Configuration
  EXPECT_CALL(configMock, getBimFilePath()).Times(AtLeast(1)).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test.bim")));
  EXPECT_CALL(configMock, getFamFilePath()).Times(AtLeast(1)).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test.fam")));
  EXPECT_CALL(configMock, getEnvironmentFilePath()).Times(AtLeast(1)).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test_env.txt")));

  EXPECT_CALL(configMock, getBedFilePath()).Times(AtLeast(1)).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test_bed.txt")));

  EXPECT_CALL(configMock, getEnvironmentIndividualIdColumnName()).Times(AtLeast(1)).WillRepeatedly(Return("indid"));
  EXPECT_CALL(configMock, getEnvironmentDelimiter()).Times(AtLeast(1)).WillRepeatedly(Return("\t "));

  EXPECT_CALL(configMock, covariateFileSpecified()).Times(AtLeast(1)).WillRepeatedly(Return(false));
  EXPECT_CALL(configMock, getPhenotypeCoding()).Times(AtLeast(1)).WillRepeatedly(Return(ONE_TWO_CODING));
  EXPECT_CALL(configMock, getGeneticModel()).Times(AtLeast(1)).WillRepeatedly(Return(DOMINANT));
  EXPECT_CALL(configMock, excludeSNPsWithNegativePosition()).Times(AtLeast(1)).WillRepeatedly(Return(false));
  EXPECT_CALL(configMock, getMinorAlleleFrequencyThreshold()).Times(AtLeast(1)).WillRepeatedly(Return(0));
}

void BedReaderThreadIntegrationTest::TearDown() {

}

TEST_F(BedReaderThreadIntegrationTest, ThreadsReadSNP) {
  const int numberOfThreads = 4;
  const int numberOfReads = 10;

  std::vector<std::vector<Container::SNPVector*>*> snpVectorResults(numberOfThreads);
  std::vector<std::thread*> threadVector(numberOfThreads);

  FileIO::DataFilesReaderFactory dataFilesReaderFactory;
  FileIO::DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configMock);

  PersonHandler* personHandler = dataFilesReader->readPersonInformation();

  EnvironmentFactorHandler* environmentFactorHandler = dataFilesReader->readEnvironmentFactorInformation(
      *personHandler);

  std::vector<SNP*>* snpInformation = dataFilesReader->readSNPInformation();
  const int numberOfSNPs = snpInformation->size();

  AlleleStatisticsFactory alleleStatisticsFactory;
  Container::SNPVectorFactory snpVectorFactory(configMock);
  FileIO::BedReader bedReader(configMock, snpVectorFactory, alleleStatisticsFactory, *personHandler, numberOfSNPs);

  for(int i = 0; i < numberOfThreads; ++i){
    std::vector<Container::SNPVector*>* snpVectors = new std::vector<Container::SNPVector*>(numberOfReads);
    snpVectorResults[i] = snpVectors;

    std::thread* t = new std::thread(CuEira::CuEira_Test::threadBedReaderTest, &bedReader, (*snpInformation)[0],
        snpVectors, numberOfReads);
    threadVector[i] = t;
  }

  for(int i = 0; i < numberOfThreads; ++i){
    threadVector[i]->join();
  }

  //Check results
  for(int thread = 0; thread < numberOfThreads; ++thread){
    std::vector<Container::SNPVector*>* snpVectors = snpVectorResults[thread];
    for(int depth = 0; depth < numberOfReads; ++depth){
      Container::SNPVector* snpVector = (*snpVectors)[depth];
      const std::vector<int>& orgSNPData = snpVector->getOrginalData();

      ASSERT_EQ(9, orgSNPData.size()); //Person 6 gets skipped because missing phenotype

      EXPECT_EQ(0, orgSNPData[0]);
      EXPECT_EQ(1, orgSNPData[1]);
      EXPECT_EQ(2, orgSNPData[2]);
      EXPECT_EQ(2, orgSNPData[3]);
      EXPECT_EQ(1, orgSNPData[4]);
      EXPECT_EQ(2, orgSNPData[5]);
      EXPECT_EQ(2, orgSNPData[6]);
      EXPECT_EQ(1, orgSNPData[7]);
      EXPECT_EQ(1, orgSNPData[8]);

      delete snpVector;
    }
    delete snpVectors;
  }

  for(int i = 0; i < numberOfThreads; ++i){
    delete threadVector[i];
  }
  for(int i = 0; i < numberOfSNPs; ++i){
    delete (*snpInformation)[i];
  }
  delete snpInformation;
  delete personHandler;
  delete environmentFactorHandler;
  delete dataFilesReader;
}

}
/* namespace FileIO */
} /* namespace CuEira */

