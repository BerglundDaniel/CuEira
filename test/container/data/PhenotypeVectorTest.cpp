#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <PhenotypeVector.h>
#include <RegularHostVector.h>
#include <PhenotypeHandlerMock.h>
#include <MissingDataHandlerMock.h>
#include <InvalidState.h>
#include <InvalidArgument.h>

using testing::Return;
using testing::ReturnRef;
using testing::_;

namespace CuEira {
namespace Container {

/**
 * Test for testing PhenotypeVector
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PhenotypeVectorTest: public ::testing::Test {
protected:
  PhenotypeVectorTest();
  virtual ~PhenotypeVectorTest();
  virtual void SetUp();
  virtual void TearDown();

};

PhenotypeVectorTest::PhenotypeVectorTest() {

}

PhenotypeVectorTest::~PhenotypeVectorTest() {

}

void PhenotypeVectorTest::SetUp() {

}

void PhenotypeVectorTest::TearDown() {

}

#ifdef DEBUG
TEST_F(PhenotypeVectorTest, Exception){
  const int numberOfIndviduals = 10;
  PhenotypeHandlerMock<RegularHostVector> phenotypeHandlerMock;

  RegularHostVector phenotypeData(numberOfIndviduals);

  EXPECT_CALL(phenotypeHandlerMock, getPhenotypeData()).Times(1).WillRepeatedly(ReturnRef(phenotypeData));
  EXPECT_CALL(phenotypeHandlerMock, getNumberOfIndividuals()).Times(1).WillRepeatedly(Return(numberOfIndviduals));

  PhenotypeVector<RegularHostVector> phenotypeVector(phenotypeHandlerMock);

  EXPECT_THROW(phenotypeVector.getPhenotypeData(), InvalidState);
  EXPECT_THROW(phenotypeVector.getNumberOfIndividualsToInclude(), InvalidState);
}
#endif

TEST_F(PhenotypeVectorTest, UpdateAndGet) {
  const int numberOfIndviduals = 10;
  const int numberOfIndvidualsExMissing = 5;
  PhenotypeHandlerMock<RegularHostVector> phenotypeHandlerMock;

  RegularHostVector phenotypeData(numberOfIndviduals);

  EXPECT_CALL(phenotypeHandlerMock, getPhenotypeData()).Times(1).WillRepeatedly(ReturnRef(phenotypeData));
  EXPECT_CALL(phenotypeHandlerMock, getNumberOfIndividuals()).Times(1).WillRepeatedly(Return(numberOfIndviduals));

  PhenotypeVector<RegularHostVector> phenotypeVector(phenotypeHandlerMock);

  ASSERT_EQ(numberOfIndviduals, phenotypeVector.getNumberOfIndividualsToInclude());

  //ApplyMissing
  MissingDataHandlerMock<RegularHostVector> missingDataHandlerMock;
  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillRepeatedly(
      Return(numberOfIndvidualsExMissing));
  EXPECT_CALL(missingDataHandlerMock, copyNonMissing(phenotypeData,_)).Times(1); //FIXME
  phenotypeVector.applyMissing(missingDataHandlerMock);

  EXPECT_EQ(numberOfIndvidualsExMissing, phenotypeVector.getNumberOfIndividualsToInclude);

  const RegularHostVector& vectorExMissing = phenotypeVector.getPhenotypeData();
  EXPECT_EQ(numberOfIndvidualsExMissing, vectorExMissing.getNumberOfRows());

  //ApplyMissingNoMissing
  phenotypeVector.applyMissing();

  EXPECT_EQ(numberOfIndviduals, phenotypeVector.getNumberOfIndividualsToInclude);

  const RegularHostMatrix& vectorIncMissing = phenotypeVector.getPhenotypeData();
  EXPECT_EQ(&phenotypeData, &vectorIncMissing);
}

} /* namespace Container */
} /* namespace CuEira */

