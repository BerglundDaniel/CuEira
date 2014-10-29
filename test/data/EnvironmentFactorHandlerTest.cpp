#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>

#include <EnvironmentFactorHandler.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <EnvironmentFactor.h>
#include <VariableType.h>
#include <Id.h>
#include <RegularHostMatrix.h>

namespace CuEira {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class EnvironmentFactorHandlerTest: public ::testing::Test {
protected:
  EnvironmentFactorHandlerTest();
  virtual ~EnvironmentFactorHandlerTest();
  virtual void SetUp();
  virtual void TearDown();

  const int numberOfIndividuals;
  const int numberOfColumns;
  Container::HostMatrix* dataMatrix;
};

EnvironmentFactorHandlerTest::EnvironmentFactorHandlerTest() :
    numberOfIndividuals(10), numberOfColumns(2), dataMatrix(nullptr) {

}

EnvironmentFactorHandlerTest::~EnvironmentFactorHandlerTest() {

}

void EnvironmentFactorHandlerTest::SetUp() {
  dataMatrix = new Container::RegularHostMatrix(numberOfIndividuals, numberOfColumns);

  //Col 1
  for(int i = 0; i < numberOfIndividuals; ++i){
    if(i > 2 && i < 6){
      (*dataMatrix)(i, 0) = 0;
    }else{
      (*dataMatrix)(i, 0) = 1;
    }
  }

  //Col 2
  for(int i = 0; i < numberOfIndividuals; ++i){
    (*dataMatrix)(i, 1) = i;
  }
}

void EnvironmentFactorHandlerTest::TearDown() {

}

TEST_F(EnvironmentFactorHandlerTest, FactorVariableTypes) {
  std::vector<EnvironmentFactor*>* environmentFactors = new std::vector<EnvironmentFactor*>(numberOfColumns);
  (*environmentFactors)[0] = new EnvironmentFactor(Id("env1"));
  (*environmentFactors)[1] = new EnvironmentFactor(Id("env2"));

  EnvironmentFactorHandler environmentFactorHandler(dataMatrix, environmentFactors);

  const std::vector<const EnvironmentFactor*>& envFactorsOut = environmentFactorHandler.getHeaders();

  EXPECT_EQ(BINARY, envFactorsOut[0]->getVariableType());
  EXPECT_EQ(OTHER, envFactorsOut[1]->getVariableType());

  for(int i = 0; i < numberOfColumns; ++i){
    EXPECT_EQ(*(*environmentFactors)[i], *(envFactorsOut[i]));
  }
}

TEST_F(EnvironmentFactorHandlerTest, Data) {
  std::vector<EnvironmentFactor*>* environmentFactors = new std::vector<EnvironmentFactor*>(numberOfColumns);
  (*environmentFactors)[0] = new EnvironmentFactor(Id("env1"));
  (*environmentFactors)[1] = new EnvironmentFactor(Id("env2"));

  EnvironmentFactorHandler environmentFactorHandler(dataMatrix, environmentFactors);

  const Container::HostVector* env1Data = environmentFactorHandler.getData(*(*environmentFactors)[0]);
  ASSERT_EQ(numberOfIndividuals, env1Data->getNumberOfRows());

  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*dataMatrix)(i, 0), (*env1Data)(i));
  }

  const Container::HostVector* env2Data = environmentFactorHandler.getData(*(*environmentFactors)[1]);
  ASSERT_EQ(numberOfIndividuals, env2Data->getNumberOfRows());

  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ((*dataMatrix)(i, 1), (*env2Data)(i));
  }

  delete env1Data;
  delete env2Data;
}

} /* namespace CuEira */

