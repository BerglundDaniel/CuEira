#include "CpuLogisticRegressionConfiguration.h"

namespace CuEira {
namespace Model {
namespace LogisticRegression {
namespace CPU {

CpuLogisticRegressionConfiguration::CpuLogisticRegressionConfiguration(const Configuration& configuration,
    const HostVector& outcomes) :
    LogisticRegressionConfiguration(configuration, false, outcomes.getNumberOfRows(), 4), outcomes(
        &outcomes), predictors(new RegularHostMatrix(numberOfRows, numberOfPredictors)), probabilites(
        new RegularHostVector(numberOfRows)), workMatrixNxM(new RegularHostMatrix(numberOfRows, numberOfPredictors)), workVectorNx1(
        new RegularHostVector(numberOfRows)), defaultBetaCoefficents(new RegularHostVector(numberOfPredictors)), scoresHost(
        new RegularHostVector(numberOfPredictors)), predictorsMemoryPointer(predictors->getMemoryPointer()) {

  setIntercept();
  setDefaultBeta(*defaultBetaCoefficents);
}

CpuLogisticRegressionConfiguration::CpuLogisticRegressionConfiguration(const Configuration& configuration,
    const HostVector& outcomes, const HostMatrix& covariates) :
    LogisticRegressionConfiguration(configuration, true, outcomes.getNumberOfRows(),
        4 + covariates.getNumberOfColumns()), outcomes(&outcomes), predictors(
        new RegularHostMatrix(numberOfRows, numberOfPredictors)), probabilites(new RegularHostVector(numberOfRows)), workMatrixNxM(
        new RegularHostMatrix(numberOfRows, numberOfPredictors)), workVectorNx1(new RegularHostVector(numberOfRows)), defaultBetaCoefficents(
        new RegularHostVector(numberOfPredictors)), scoresHost(new RegularHostVector(numberOfPredictors)), predictorsMemoryPointer(
        predictors->getMemoryPointer()) {

  setIntercept();
  setDefaultBeta(*defaultBetaCoefficents);
  setCovariates(covariates);
}

CpuLogisticRegressionConfiguration::CpuLogisticRegressionConfiguration(const Configuration& configuration) :
    LogisticRegressionConfiguration(configuration, false, 0, 0), outcomes(nullptr), predictors(nullptr), probabilites(
        nullptr), workMatrixNxM(nullptr), workVectorNx1(nullptr), defaultBetaCoefficents(nullptr), scoresHost(nullptr), predictorsMemoryPointer(
        nullptr) {

}

CpuLogisticRegressionConfiguration::~CpuLogisticRegressionConfiguration() {
  delete predictors;
  delete probabilites;
  delete workMatrixNxM;
  delete workVectorNx1;
  delete defaultBetaCoefficents;
  delete scoresHost;
}

void CpuLogisticRegressionConfiguration::setIntercept() {
  for(int i = 0; i < numberOfRows; ++i){
    (*predictors)(i, 0) = 1;
  }
}

void CpuLogisticRegressionConfiguration::setCovariates(const HostMatrix& covariates) {
  for(int i = 0; i < covariates.getNumberOfColumns(); ++i){
    const HostVector* covColumn = covariates(i);
    HostVector* covPredictorColumn = (*predictors)(4 + i);

    Blas::copyVector(*covColumn, *covPredictorColumn);

    delete covPredictorColumn;
    delete covColumn;
  }

}

void CpuLogisticRegressionConfiguration::setEnvironmentFactor(const HostVector& environmentData) {
  //Putting the environment as the third column
  HostVector* envPredictorColumn = (*predictors)(2);
  Blas::copyVector(environmentData, *envPredictorColumn);
  delete envPredictorColumn;
}

void CpuLogisticRegressionConfiguration::setSNP(const HostVector& snpData) {
  //Putting the snp column as the second column
  HostVector* snpPredictorColumn = (*predictors)(1);
  Blas::copyVector(snpData, *snpPredictorColumn);
  delete snpPredictorColumn;
}

void CpuLogisticRegressionConfiguration::setInteraction(const HostVector& interactionVector) {
  //Putting the interaction column as the fourth column
  HostVector* interactionPredictorColumn = (*predictors)(3);
  Blas::copyVector(interactionVector, *interactionPredictorColumn);
  delete interactionPredictorColumn;
}

HostVector& CpuLogisticRegressionConfiguration::getProbabilites() {
  return *probabilites;
}

HostVector& CpuLogisticRegressionConfiguration::getScoresHost() {
  return *scoresHost;
}

const HostMatrix& CpuLogisticRegressionConfiguration::getPredictors() const {
  return *predictors;
}

const HostVector& CpuLogisticRegressionConfiguration::getOutcomes() const {
  return *outcomes;
}

HostMatrix& CpuLogisticRegressionConfiguration::getWorkMatrixNxM() {
  return *workMatrixNxM;
}

HostVector& CpuLogisticRegressionConfiguration::getWorkVectorNx1() {
  return *workVectorNx1;
}

const HostVector& CpuLogisticRegressionConfiguration::getDefaultBetaCoefficents() const {
  return *defaultBetaCoefficents;
}

} /* namespace CPU */
} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
