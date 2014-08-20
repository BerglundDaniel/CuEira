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

  for(int i = 0; i < numberOfSNPs; ++i){
    std::ostringstream os;
    os << "snp" << i;
    Id id(os.str());

    SNP* snp = new SNP(id, "allele1", "allele2", 1);
    (*snpQueue)[i] = snp;
  }

  Task::DataQueue dataQueue(*snpQueue);

  for(int i = numberOfSNPs - 1; i >= 0; --i){
    SNP* snp = dataQueue.next();
    ASSERT_TRUE(snp != nullptr);

    EXPECT_EQ(*((*snpQueue)[i]), *snp);
    delete snp;
  }
  SNP* snp = dataQueue.next();
  ASSERT_TRUE(snp == nullptr);

  delete snpQueue;
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */
