#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>

#include <DataQueue.h>
#include <SNP.h>

namespace CuEira {
namespace CuEira_Test {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataQueueTest: public ::testing::Test {
protected:
  DataQueueTest();
  virtual ~DataQueueTest();
  virtual void SetUp();
  virtual void TearDown();

};

DataQueueTest::DataQueueTest() {

}

DataQueueTest::~DataQueueTest() {

}

void DataQueueTest::SetUp() {

}

void DataQueueTest::TearDown() {

}

TEST_F(DataQueueTest, QueueTest) {
  const int numberOfSNPs = 3;
  std::vector<SNP*>* snpQueue = new std::vector<SNP*>(numberOfSNPs);
  std::vector<SNP*>* snpVector = new std::vector<SNP*>(numberOfSNPs);

  for(int i = 0; i < numberOfSNPs; ++i){
    std::ostringstream os;
    os << "snp" << i;
    Id id(os.str());

    SNP* snp = new SNP(id, "allele1", "allele2", 1);
    (*snpQueue)[i] = snp;
    (*snpVector)[i] = snp;
  }

  Task::DataQueue dataQueue(snpQueue);

  for(int i = 0; i < numberOfSNPs; ++i){
    EXPECT_TRUE(dataQueue.hasNext());
    SNP* snp = dataQueue.next();

    std::cerr << "asdf " << snp->getId().getString() << std::endl;

    EXPECT_TRUE(*((*snpVector)[i]) == *snp);
    EXPECT_EQ(*((*snpVector)[i]), *snp);
    delete snp;
  }
  EXPECT_FALSE(dataQueue.hasNext());

  delete snpVector;
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */
