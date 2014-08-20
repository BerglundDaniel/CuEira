#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <thread>

#include <DataQueue.h>
#include <SNP.h>

namespace CuEira {
namespace CuEira_Test {

void threadDataQueueTest(Task::DataQueue* dataQueue, std::vector<SNP*>* snpVector) {
  while(true){
    SNP* snp = dataQueue->next();
    if(snp == nullptr){
      break;
    }
    snpVector->push_back(snp);
  } //While true
}

/**
 * Test for testing DataQueue with threads
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataQueueThreadTest: public ::testing::Test {
protected:
  DataQueueThreadTest();
  virtual ~DataQueueThreadTest();
  virtual void SetUp();
  virtual void TearDown();

};

DataQueueThreadTest::DataQueueThreadTest() {

}

DataQueueThreadTest::~DataQueueThreadTest() {

}

void DataQueueThreadTest::SetUp() {

}

void DataQueueThreadTest::TearDown() {

}

TEST_F(DataQueueThreadTest, ThreadTest) {
  const int numberOfSNPs = 10;
  const int numberOfThreads = 3;
  std::vector<SNP*>* snpQueue = new std::vector<SNP*>(numberOfSNPs);

  for(int i = 0; i < numberOfSNPs; ++i){
    std::ostringstream os;
    os << "snp" << i;
    Id id(os.str());

    SNP* snp = new SNP(id, "allele1", "allele2", 1);
    (*snpQueue)[i] = snp;
  }

  Task::DataQueue* dataQueue = new Task::DataQueue(*snpQueue);
  std::vector<std::vector<SNP*>*> vectorOfSNPVector(numberOfThreads);
  std::vector<std::thread*> threadVector(numberOfThreads);

  for(int i = 0; i < numberOfThreads; ++i){
    std::vector<SNP*>* snpVector = new std::vector<SNP*>();
    vectorOfSNPVector[i] = snpVector;

    std::thread* t = new std::thread(CuEira::CuEira_Test::threadDataQueueTest, dataQueue, snpVector);
    threadVector[i] = t;
  }

  for(int i = 0; i < numberOfThreads; ++i){
    threadVector[i]->join();
  }

  //Make sure things are correct
  std::vector<bool> accessedSnps(numberOfSNPs);
  for(int i = 0; i < numberOfSNPs; ++i){
    accessedSnps[i] = false;
  }

  for(int i = 0; i < numberOfThreads; ++i){
    std::vector<SNP*>* snpVector = vectorOfSNPVector[i];
    const int size = snpVector->size();

    for(int j = 0; j < size; ++j){
      SNP* snp = (*snpVector)[j];

      for(int k = 0; k < numberOfSNPs; ++k){
        if(*(*snpQueue)[k] == *snp){
          ASSERT_FALSE(accessedSnps[k]);
          accessedSnps[k] = true;
          break;
        }
      } //for k
    } //for j
  } //for i

  for(int i = 0; i < numberOfSNPs; ++i){
    ASSERT_TRUE(accessedSnps[i]);
  }

  //Delete stuff
  for(int i = 0; i < numberOfThreads; ++i){
    delete vectorOfSNPVector[i];
    delete threadVector[i];
  }

  for(int i = 0; i < numberOfSNPs; ++i){
    delete (*snpQueue)[i];
  }

  delete snpQueue;
  delete dataQueue;
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */
