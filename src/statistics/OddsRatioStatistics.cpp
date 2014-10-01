#include "OddsRatioStatistics.h"

namespace CuEira {

OddsRatioStatistics::OddsRatioStatistics(
    const Model::LogisticRegression::LogisticRegressionResult* logisticRegressionResult) :
    logisticRegressionResult(logisticRegressionResult), betaCoefficents(&logisticRegressionResult->getBeta()), standardError(
        calculateStandardError(logisticRegressionResult->getInverseInformationMatrix())), oddsRatios(
        calculateOddsRatios(*betaCoefficents)), oddsRatiosLow(calculateOddsRatiosLow(*betaCoefficents, *standardError)), oddsRatiosHigh(
        calculateOddsRatiosHigh(*betaCoefficents, *standardError)) {

}

OddsRatioStatistics::OddsRatioStatistics() :
    logisticRegressionResult(nullptr), standardError(nullptr), oddsRatios(nullptr), oddsRatiosLow(nullptr), oddsRatiosHigh(
        nullptr), betaCoefficents(nullptr) {

}

OddsRatioStatistics::~OddsRatioStatistics() {
  delete logisticRegressionResult;
  delete standardError;
  delete oddsRatios;
  delete oddsRatiosLow;
  delete oddsRatiosHigh;
}

const std::vector<double>& OddsRatioStatistics::getOddsRatios() const {
  return *oddsRatios;
}

const std::vector<double>& OddsRatioStatistics::getOddsRatiosLow() const {
  return *oddsRatiosLow;
}

const std::vector<double>& OddsRatioStatistics::getOddsRatiosHigh() const {
  return *oddsRatiosHigh;
}

std::vector<double>* OddsRatioStatistics::calculateStandardError(const Container::HostMatrix& covarianceMatrix) const {
  const int size = covarianceMatrix.getNumberOfRows(); //Symmetrical matrix
  std::vector<double>* standardError = new std::vector<double>(size);

  for(int i = 0; i < size; ++i){
    (*standardError)[i] = covarianceMatrix(i, i);
  }

  return standardError;
}

std::vector<double>* OddsRatioStatistics::calculateOddsRatios(const Container::HostVector& betaCoefficents) const {
  const int size = betaCoefficents.getNumberOfRows() - 1; //Skipping the intercept
  std::vector<double>* oddsRatios = new std::vector<double>(size);

  for(int i = 0; i < size; ++i){
    (*oddsRatios)[i] = exp(betaCoefficents(i + 1));
  }

  return oddsRatios;
}

std::vector<double>* OddsRatioStatistics::calculateOddsRatiosLow(const Container::HostVector& betaCoefficents,
    const std::vector<double>& standardError) const {

  const int size = betaCoefficents.getNumberOfRows() - 1; //Skipping the intercept
  std::vector<double>* oddsRatiosLow = new std::vector<double>(size);

  for(int i = 0; i < size; ++i){
    (*oddsRatiosLow)[i] = exp(-1.96 * standardError[i + 1] + betaCoefficents(i + 1));
  }

  return oddsRatiosLow;
}

std::vector<double>* OddsRatioStatistics::calculateOddsRatiosHigh(const Container::HostVector& betaCoefficents,
    const std::vector<double>& standardError) const {

  const int size = betaCoefficents.getNumberOfRows() - 1; //Skipping the intercept
  std::vector<double>* oddsRatiosHigh = new std::vector<double>(size);

  for(int i = 0; i < size; ++i){
    (*oddsRatiosHigh)[i] = exp(1.96 * standardError[i + 1] + betaCoefficents(i + 1));
  }

  return oddsRatiosHigh;
}

void OddsRatioStatistics::toOstream(std::ostream& os) const {
  //Print ORs including covariates if any
  const int size = betaCoefficents->getNumberOfRows() - 1; //Skipping the intercept
  for(int i = 0; i < size - 1; ++i){
    os << (*(oddsRatios))[i] << "," << (*(oddsRatiosLow))[i] << "," << (*(oddsRatiosHigh))[i] << ",";
  }
  os << (*(oddsRatios))[size - 1] << "," << (*(oddsRatiosLow))[size - 1] << "," << (*(oddsRatiosHigh))[size - 1];
}

std::ostream & operator<<(std::ostream& os, const OddsRatioStatistics& oddsRatioStatistics) {
  oddsRatioStatistics.toOstream(os);
  return os;
}

} /* namespace CuEira */
