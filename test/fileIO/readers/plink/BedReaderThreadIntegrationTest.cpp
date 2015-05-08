#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <thread>
#include <utility>

#include <BedReader.h>
#include <ConfigurationMock.h>
#include <PersonHandlerMock.h>
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
#include <CpuSNPVectorFactory.h>

using testing::Return;
using testing::_;
using testing::ReturnRef;
using testing::Eq;
using testing::DoAll;
using testing::AtLeast;

namespace CuEira {

namespace CuEira_Test {

void threadBedReaderTest(FileIO::BedReader* bedReader, SNP* snp, std::vector<Container::SNPVector*>* snpVectors,
    const int numberOfReads) {
  for(int i = 0; i < numberOfReads; ++i){
    (*snpVectors)[i] = bedReader->readSNP(*snp);
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
  EXPECT_CALL(configMock, getBedFilePath()).Times(AtLeast(1)).WillRepeatedly(
      Return(std::string(CuEira_BUILD_DIR) + std::string("/test.bed")));
  EXPECT_CALL(configMock, getGeneticModel()).Times(AtLeast(1)).WillRepeatedly(Return(DOMINANT));
}

void BedReaderThreadIntegrationTest::TearDown() {

}

TEST_F(BedReaderThreadIntegrationTest, ThreadsReadSNP) {
  const int numberOfThreads = 4;
  const int numberOfReads = 10;
  const int numberOfSNPs = 10;
  const int numberOfIndividualsTotal = 10;

  std::vector<std::vector<Container::SNPVector*>*> snpVectorResults(numberOfThreads);
  std::vector<std::thread*> threadVector(numberOfThreads);
  SNP* snp = new SNP(Id("snp1"), "a1", "a2", 0);

  std::vector<Person*> persons(numberOfIndividualsTotal);
  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    std::ostringstream os;
    os << "ind" << i;
    Id id(os.str());

    if(i == 6){
      persons[i] = new Person(id, MALE, AFFECTED, false);
    }else{
      persons[i] = new Person(id, MALE, AFFECTED, true);
    }
  }

  Container::SNPVectorFactory* snpVectorFactory = new Container::CPU::CpuSNPVectorFactory(configMock);
  PersonHandlerMock personHandlerMock;

  EXPECT_CALL(personHandlerMock, getNumberOfIndividualsTotal()).Times(1).WillOnce(Return(numberOfIndividualsTotal));

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_CALL(personHandlerMock, getPersonFromRowAll(i)).WillRepeatedly(ReturnRef(persons[i]));
  }

  FileIO::BedReader bedReader(configMock, snpVectorFactory, personHandlerMock, numberOfSNPs);

  for(int i = 0; i < numberOfThreads; ++i){
    std::vector<Container::SNPVector*>* snpVectors = new std::vector<Container::SNPVector*>(numberOfReads);
    snpVectorResults[i] = snpVectors;

    std::thread* t = new std::thread(CuEira::CuEira_Test::threadBedReaderTest, &bedReader, snp, snpVectors,
        numberOfReads);
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
      const std::vector<int>& orgSNPData = snpVector->getOriginalSNPData();

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
  delete snp;
}

}
/* namespace FileIO */
} /* namespace CuEira */

