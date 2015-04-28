#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>

#include <CovariatesMatrix.h>
#include <RegularHostVector.h>
#include <RegularHostMatrix.h>
#include <CovariatesHandlerMock.h>
#include <InvalidState.h>
#include <MissingDataHandlerMock.h>

using testing::Return;
using testing::ReturnRef;
using testing::_;

namespace CuEira {
namespace Container {

using namespace CuEira::CPU;

/**
 * Test for testing CovariatesMatrix using the CPU templates
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuCovariatesMatrixTest: public ::testing::Test {
protected:
  CpuCovariatesMatrixTest();
  virtual ~CpuCovariatesMatrixTest();
  virtual void SetUp();
  virtual void TearDown();
};

CpuCovariatesMatrixTest::CpuCovariatesMatrixTest() {

}

CpuCovariatesMatrixTest::~CpuCovariatesMatrixTest() {

}

void CpuCovariatesMatrixTest::SetUp() {

}

void CpuCovariatesMatrixTest::TearDown() {

}

#ifdef DEBUG
TEST_F(CpuCovariatesMatrixTest, Exception){
  const int numberOfRows = 3;
  const int numberOfCovariates = 5;
  CovariatesHandlerMock<RegularHostMatrix> covariatesHandlerMock;
  RegularHostMatrix covariates(rows, numberOfCovariates);

  EXPECT_CALL(covariatesHandlerMock, getCovariatesMatrix()).Times(1).WillRepeatedly(ReturnRef(covariates));
  EXPECT_CALL(covariatesHandlerMock, getNumberOfCovariates()).Times(1).WillRepeatedly(Return(numberOfCovariates));

  CovariatesMatrix<RegularHostMatrix, RegularHostVector> covariatesMatrix(covariatesHandlerMock);

  EXPECT_THROW(covariatesMatrix.getCovariatesData(), InvalidState);
}
#endif

TEST_F(CpuCovariatesMatrixTest, ApplyMissing) {
  const int numberOfRows = 10;
  const int numberOfRowsExMissing = 3;
  const int numberOfCovariates = 5;
  CovariatesHandlerMock<RegularHostMatrix> covariatesHandlerMock;
  RegularHostMatrix covariates(rows, numberOfCovariates);

  EXPECT_CALL(covariatesHandlerMock, getCovariatesMatrix()).Times(1).WillRepeatedly(ReturnRef(covariates));
  EXPECT_CALL(covariatesHandlerMock, getNumberOfCovariates()).Times(1).WillRepeatedly(Return(numberOfCovariates));

  CovariatesMatrix<RegularHostMatrix, RegularHostVector> covariatesMatrix(covariatesHandlerMock);

  //ApplyMissing
  MissingDataHandlerMock<RegularHostVector> missingDataHandlerMock;

  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillRepeatedly(
      Return(numberOfRowsExMissing));
  EXPECT_CALL(missingDataHandlerMock, copyNonMissing(_,_)).Times(numberOfCovariates);

  covariatesMatrix.applyMissing(missingDataHandlerMock);

  const RegularHostMatrix& matrixExMissing = covariatesMatrix.getCovariatesData();
  EXPECT_EQ(&covariatesMatrix.covariatesExMissing, &matrixExMissing);
  EXPECT_EQ(numberOfRowsExMissing, matrixExMissing.getNumberOfRows());

  //ApplyMissingNoMissing
  covariatesMatrix.applyMissing();
  const RegularHostMatrix& matrixIncMissing = covariatesMatrix.getCovariatesData();
  EXPECT_EQ(&covariates, &matrixIncMissing);
}

}
/* namespace Container */
} /* namespace CuEira */

