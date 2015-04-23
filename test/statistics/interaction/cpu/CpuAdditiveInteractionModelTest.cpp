#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <ostream>

#include <CpuAdditiveInteractionModel.h>
#include <EnvironmentVectorMock.h>
#include <InteractionVectorMock.h>
#include <SNPVectorMock.h>
#include <RegularHostVector.h>
#include <MKLWrapper.h>

using testing::Return;
using testing::ReturnRef;
using testing::AtLeast;

namespace CuEira {
namespace CPU {

/**
 * Test for testing CpuAdditiveInteractionModel
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuAdditiveInteractionModelTest: public ::testing::Test {
protected:
  CpuAdditiveInteractionModelTest();
  virtual ~CpuAdditiveInteractionModelTest();
  virtual void SetUp();
  virtual void TearDown();

};

CpuAdditiveInteractionModelTest::CpuAdditiveInteractionModelTest() {

}

CpuAdditiveInteractionModelTest::~CpuAdditiveInteractionModelTest() {

}

void CpuAdditiveInteractionModelTest::SetUp() {

}

void CpuAdditiveInteractionModelTest::TearDown() {

}

TEST_F(CpuAdditiveInteractionModelTest, Construct) {
  const int numberOfIndividuals = 10;
  MKLWrapper& mklWrapper;
  CpuAdditiveInteractionModel cpuAdditiveInteractionModel(mklWrapper);

  Container::RegularHostVector snpData(numberOfIndividuals);
  Container::RegularHostVector envData(numberOfIndividuals);
  Container::RegularHostVector snpDataOrg(numberOfIndividuals);
  Container::RegularHostVector envDataOrg(numberOfIndividuals);
  Container::RegularHostVector interData(numberOfIndividuals);

  for(int i = 0; i < numberOfIndividuals; ++i){
    snpData(i) = i % 2;
    snpDataOrg(i) = snpData(i);
  }

  for(int i = 0; i < numberOfIndividuals; ++i){
    envData(i) = (i + 2) % 3;
    envDataOrg(i) = envData(i);
  }

  Container::SNPVectorMock<Container::RegularHostVector> snpVectorMock;
  EXPECT_CALL(snpVectorMock, getNumberOfIndividualsToInclude()).WillRepeatedly(Return(numberOfIndividuals));
  EXPECT_CALL(snpVectorMock, getOriginalSNPData()).Times(1).WillRepeatedly(ReturnRef(snpData));

  Container::EnvironmentVectorMock<Container::RegularHostVector> environmentVectorMock;
  EXPECT_CALL(environmentVectorMock, getNumberOfIndividualsToInclude()).WillRepeatedly(Return(numberOfIndividuals));
  EXPECT_CALL(environmentVectorMock, getEnvironmentData()).Times(1).WillRepeatedly(ReturnRef(envData));

  Container::InteractionVectorMock<Container::RegularHostVector> interactionVectorMock;
  EXPECT_CALL(interactionVectorMock, getNumberOfIndividualsToInclude()).WillRepeatedly(Return(numberOfIndividuals));
  EXPECT_CALL(interactionVectorMock, getInteractionData()).Times(1).WillRepeatedly(ReturnRef(interData));
  EXPECT_CALL(interactionVectorMock, updateSize(numberOfIndividuals)).Times(1);

  cpuAdditiveInteractionModel.applyModel(snpVectorMock, environmentVectorMock, interactionVectorMock);

  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ(snpDataOrg(i) * envDataOrg(i), interData(i));

    if(interData(i) == 0){
      EXPECT_EQ(snpDataOrg(i), snpData(i));
      EXPECT_EQ(envDataOrg(i), envData(i));
    }else{
      EXPECT_EQ(0, snpData(i));
      EXPECT_EQ(0, envData(i));
    }
  }
}

} /* namespace CPU */
} /* namespace CuEira */
