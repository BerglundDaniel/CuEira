#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>

#include <CovariatesMatrix.h>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <CovariatesHandlerMock.h>
#include <InvalidState.h>
#include <MissingDataHandlerMock.h>

using testing::Return;
using testing::ReturnRef;
using testing::_;

namespace CuEira {
namespace Container {

using namespace CuEira::CUDA;

/**
 * Test for testing CovariatesMatrix using the CUDA templates
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaCovariatesMatrixTest: public ::testing::Test {
protected:
  CudaCovariatesMatrixTest();
  virtual ~CudaCovariatesMatrixTest();
  virtual void SetUp();
  virtual void TearDown();
};

CudaCovariatesMatrixTest::CudaCovariatesMatrixTest() {

}

CudaCovariatesMatrixTest::~CudaCovariatesMatrixTest() {

}

void CudaCovariatesMatrixTest::SetUp() {

}

void CudaCovariatesMatrixTest::TearDown() {

}

#ifdef DEBUG
TEST_F(CudaCovariatesMatrixTest, Exception){
  const int numberOfRows = 3;
  const int numberOfCovariates = 5;
  CovariatesHandlerMock<DeviceMatrix> covariatesHandlerMock;
  DeviceMatrix covariates(rows, numberOfCovariates);

  EXPECT_CALL(covariatesHandlerMock, getCovariatesMatrix()).Times(1).WillRepeatedly(ReturnRef(covariates));
  EXPECT_CALL(covariatesHandlerMock, getNumberOfCovariates()).Times(1).WillRepeatedly(Return(numberOfCovariates));

  CovariatesMatrix<DeviceMatrix, DeviceVector> covariatesMatrix(covariatesHandlerMock);

  EXPECT_THROW(covariatesMatrix.getCovariatesData(), InvalidState);
}
#endif

TEST_F(CudaCovariatesMatrixTest, ApplyMissing) {
  const int numberOfRows = 10;
  const int numberOfRowsExMissing = 3;
  const int numberOfCovariates = 5;
  CovariatesHandlerMock<DeviceMatrix> covariatesHandlerMock;
  DeviceMatrix covariates(rows, numberOfCovariates);

  EXPECT_CALL(covariatesHandlerMock, getCovariatesMatrix()).Times(1).WillRepeatedly(ReturnRef(covariates));
  EXPECT_CALL(covariatesHandlerMock, getNumberOfCovariates()).Times(1).WillRepeatedly(Return(numberOfCovariates));

  CovariatesMatrix<DeviceMatrix, DeviceVector> covariatesMatrix(covariatesHandlerMock);

  //ApplyMissing
  MissingDataHandlerMock<DeviceVector> missingDataHandlerMock;

  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillRepeatedly(
      Return(numberOfRowsExMissing));
  EXPECT_CALL(missingDataHandlerMock, copyNonMissing(_,_)).Times(numberOfCovariates);

  covariatesMatrix.applyMissing(missingDataHandlerMock);

  const DeviceMatrix& matrixExMissing = covariatesMatrix.getCovariatesData();
  EXPECT_EQ(&covariatesMatrix.covariatesExMissing, &matrixExMissing);
  EXPECT_EQ(numberOfRowsExMissing, matrixExMissing.getNumberOfRows());

  //ApplyMissingNoMissing
  covariatesMatrix.applyMissing();
  const DeviceMatrix& matrixIncMissing = covariatesMatrix.getCovariatesData();
  EXPECT_EQ(&covariates, &matrixIncMissing);
}

}
/* namespace Container */
} /* namespace CuEira */

