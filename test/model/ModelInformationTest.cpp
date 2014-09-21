#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <ostream>

#include <ModelInformation.h>
#include <ModelState.h>

namespace CuEira {
namespace Model {

/**
 * Test for testing ModelInformation
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ModelInformationTest: public ::testing::Test {
protected:
  ModelInformationTest();
  virtual ~ModelInformationTest();
  virtual void SetUp();
  virtual void TearDown();

};

ModelInformationTest::ModelInformationTest() {

}

ModelInformationTest::~ModelInformationTest() {

}

void ModelInformationTest::SetUp() {

}

void ModelInformationTest::TearDown() {

}

TEST_F(ModelInformationTest, ModelState) {
  for(int i = 0; i < 3; ++i){
    ModelState state = (ModelState)i;
    ModelInformation modelInformation(state, "");
    EXPECT_EQ(state, modelInformation.getModelState());
  }

}

TEST_F(ModelInformationTest, Ostream) {
  std::string str("asdf");
  ModelInformation modelInformation(DONE, str);

  std::ostringstream os;
  os << modelInformation;

  EXPECT_EQ(str, os.str());
}

}
/* namespace Model */
} /* namespace CuEira */
