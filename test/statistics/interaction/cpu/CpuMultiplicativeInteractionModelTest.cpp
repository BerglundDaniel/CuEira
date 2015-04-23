#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <ostream>

#include <CpuMultiplicativeInteractionModel.h>
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
 * Test for testing CpuMultiplicativeInteractionModel
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuMultiplicativeInteractionModelTest: public ::testing::Test {
protected:
  CpuMultiplicativeInteractionModelTest();
  virtual ~CpuMultiplicativeInteractionModelTest();
  virtual void SetUp();
  virtual void TearDown();

};

CpuMultiplicativeInteractionModelTest::CpuMultiplicativeInteractionModelTest() {

}

CpuMultiplicativeInteractionModelTest::~CpuMultiplicativeInteractionModelTest() {

}

void CpuMultiplicativeInteractionModelTest::SetUp() {

}

void CpuMultiplicativeInteractionModelTest::TearDown() {

}

TEST_F(CpuMultiplicativeInteractionModelTest, Construct) {
  const int numberOfIndividuals = 10;
  MKLWrapper& mklWrapper;
  CpuMultiplicativeInteractionModel cpuMultiplicativeInteractionModel(mklWrapper);

  Container::RegularHostVector snpData(numberOfIndividuals);
  Container::RegularHostVector envData(numberOfIndividuals);
  Container::RegularHostVector interData(numberOfIndividuals);

  for(int i = 0; i < numberOfIndividuals; ++i){
    snpData(i) = i;
  }

  for(int i = 0; i < numberOfIndividuals; ++i){
    envData(i) = numberOfIndividuals - i;
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

  cpuMultiplicativeInteractionModel.applyModel(snpVectorMock, environmentVectorMock, interactionVectorMock);

  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ(i * (numberOfIndividuals - i), interData(i));
  }
}

} /* namespace CPU */
} /* namespace CuEira */
