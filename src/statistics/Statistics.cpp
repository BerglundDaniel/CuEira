#include "Statistics.h"

namespace CuEira {

Statistics::Statistics(const Model::LogisticRegression::LogisticRegressionResult* logisticRegressionResult) :
    logisticRegressionResult(logisticRegressionResult), betaCoefficents(logisticRegressionResult->getBeta()), standardError(
        calculateStandardError(logisticRegressionResult->getInverseInformationMatrix())), ap(
        calculateAp(reri, betaCoefficents(3))), reri(calculateReri(*oddsRatios)), oddsRatios(
        calculateOddsRatios(betaCoefficents)), oddsRatiosLow(calculateOddsRatiosLow(betaCoefficents, *standardError)), oddsRatiosHigh(
        calculateOddsRatiosHigh(betaCoefficents, *standardError)) {

}

Statistics::~Statistics() {
  delete logisticRegressionResult;
  delete standardError;
  delete oddsRatios;
  delete oddsRatiosLow;
  delete oddsRatiosHigh;
}

double Statistics::getReri() const {
  return reri;
}

double Statistics::getAp() const {
  return ap;
}

const std::vector<double>& Statistics::getOddsRatios() const {
  return *oddsRatios;
}

const std::vector<double>& Statistics::getOddsRatiosLow() const {
  return *oddsRatiosLow;
}

const std::vector<double>& Statistics::getOddsRatiosHigh() const {
  return *oddsRatiosHigh;
}

double Statistics::calculateReri(const std::vector<double>& oddsRatios) const {
  return oddsRatios[2] - oddsRatios[0] - oddsRatios[1] + 1;
}

double Statistics::calculateAp(double reri, PRECISION interactionBeta) const {
  return reri / interactionBeta;
}

std::vector<double>* Statistics::calculateStandardError(const Container::HostMatrix& covarianceMatrix) const {
  const int size = covarianceMatrix.getNumberOfRows(); //Symmetrical matrix
  std::vector<double>* standardError = new std::vector<double>(size);

  for(int i = 0; i < size; ++i){
    (*standardError)[i] = covarianceMatrix(i, i);
  }

  return standardError;
}

std::vector<double>* Statistics::calculateOddsRatios(const Container::HostVector& betaCoefficents) const {
  const int size = betaCoefficents.getNumberOfRows() - 1; //Skipping the intercept
  std::vector<double>* oddsRatios = new std::vector<double>(size);

  for(int i = 0; i < size; ++i){
    (*oddsRatios)[i] = exp(betaCoefficents(i + 1));
  }

  return oddsRatios;
}

std::vector<double>* Statistics::calculateOddsRatiosLow(const Container::HostVector& betaCoefficents,
    const std::vector<double>& standardError) const {

  const int size = betaCoefficents.getNumberOfRows() - 1; //Skipping the intercept
  std::vector<double>* oddsRatiosLow = new std::vector<double>(size);

  for(int i = 0; i < size; ++i){
    (*oddsRatiosLow)[i] = exp(-1.96 * standardError[i + 1] + betaCoefficents(i + 1));
  }

  return oddsRatiosLow;
}

std::vector<double>* Statistics::calculateOddsRatiosHigh(const Container::HostVector& betaCoefficents,
    const std::vector<double>& standardError) const {

  const int size = betaCoefficents.getNumberOfRows() - 1; //Skipping the intercept
  std::vector<double>* oddsRatiosHigh = new std::vector<double>(size);

  for(int i = 0; i < size; ++i){
    (*oddsRatiosHigh)[i] = exp(1.96 * standardError[i + 1] + betaCoefficents(i + 1));
  }

  return oddsRatiosHigh;
}

std::ostream & operator<<(std::ostream& os, const Statistics& statistics) {
  //Print AP
  os << statistics.ap << ", ";

  //Print RERIR
  os << statistics.reri << ", ";

  //Print ORs including covariates if any
  const int size = statistics.betaCoefficents.getNumberOfRows() - 1; //Skipping the intercept
  for(int i = 0; i < size - 1; ++i){
    os << (*(statistics.oddsRatios))[i] << ", " << (*(statistics.oddsRatiosLow))[i] << ", "
        << (*(statistics.oddsRatiosHigh))[i] << ", ";
  }
  os << (*(statistics.oddsRatios))[size - 1] << ", " << (*(statistics.oddsRatiosLow))[size - 1] << ", "
      << (*(statistics.oddsRatiosHigh))[size - 1];

  return os;
}

} /* namespace CuEira */
