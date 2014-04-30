#include <gmock/gmock.h>
using ::testing::Eq;
#include <gtest/gtest.h>
using ::testing::Test;

#include <Person.h>
#include <PersonHandler.h>

#define private public

namespace CuEira {
namespace Testing {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PersonHandlerTest: public Test {
protected:
  PersonHandlerTest() {
  }
  ~PersonHandlerTest() {
  }

  virtual void SetUp() {
  }
  virtual void TearDown() {
  }

  ToDo list;

  static const size_t taskCount = 3;
  static const string tasks[taskCount];
};

const string PersonHandlerTest::tasks[taskCount] = {"write code", "compile", "test"};

} /* namespace Testing */
} /* namespace CuEira */

