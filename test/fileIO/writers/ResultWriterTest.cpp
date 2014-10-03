#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <thread>

#include <ResultWriter.h>

using testing::Return;
using testing::_;
using testing::ReturnRef;
using testing::Eq;

namespace CuEira {
namespace FileIO {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ResultWriterTest: public ::testing::Test {
protected:
  ResultWriterTest();
  virtual ~ResultWriterTest();
  virtual void SetUp();
  virtual void TearDown();

};

ResultWriterTest::ResultWriterTest() {

}

ResultWriterTest::~ResultWriterTest() {

}

void ResultWriterTest::SetUp() {

}

void ResultWriterTest::TearDown() {

}

TEST_F(ResultWriterTest, Write) {
//TODO
}

}
/* namespace FileIO */
} /* namespace CuEira */

