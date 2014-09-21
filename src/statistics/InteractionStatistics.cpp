#include "InteractionStatistics.h"

namespace CuEira {

InteractionStatistics::InteractionStatistics(
    const Model::LogisticRegression::LogisticRegressionResult* logisticRegressionResult) :
    logisticRegressionResult(logisticRegressionResult), betaCoefficents(logisticRegressionResult->getBeta()), standardError(
        calculateStandardError(logisticRegressionResult->getInverseInformationMatrix())), ap(
        calculateAp(reri, betaCoefficents(3))), reri(calculateReri(*oddsRatios)), oddsRatios(
        calculateOddsRatios(betaCoefficents)), oddsRatiosLow(calculateOddsRatiosLow(betaCoefficents, *standardError)), oddsRatiosHigh(
        calculateOddsRatiosHigh(betaCoefficents, *standardError)) {

}

InteractionStatistics::~InteractionStatistics() {
  delete logisticRegressionResult;
  delete standardError;
  delete oddsRatios;
  delete oddsRatiosLow;
  delete oddsRatiosHigh;
}

double InteractionStatistics::getReri() const {
  return reri;
}

double InteractionStatistics::getAp() const {
  return ap;
}

const std::vector<double>& InteractionStatistics::getOddsRatios() const {
  return *oddsRatios;
}

const std::vector<double>& InteractionStatistics::getOddsRatiosLow() const {
  return *oddsRatiosLow;
}

const std::vector<double>& InteractionStatistics::getOddsRatiosHigh() const {
  return *oddsRatiosHigh;
}

double InteractionStatistics::calculateReri(const std::vector<double>& oddsRatios) const {
  return oddsRatios[2] - oddsRatios[0] - oddsRatios[1] + 1;
}

double InteractionStatistics::calculateAp(double reri, PRECISION interactionBeta) const {
  return reri / interactionBeta;
}

std::vector<double>* InteractionStatistics::calculateStandardError(
    const Container::HostMatrix& covarianceMatrix) const {
  const int size = covarianceMatrix.getNumberOfRows(); //Symmetrical matrix
  std::vector<double>* standardError = new std::vector<double>(size);

  for(int i = 0; i < size; ++i){
    (*standardError)[i] = covarianceMatrix(i, i);
  }

  return standardError;
}

std::vector<double>* InteractionStatistics::calculateOddsRatios(const Container::HostVector& betaCoefficents) const {
  const int size = betaCoefficents.getNumberOfRows() - 1; //Skipping the intercept
  std::vector<double>* oddsRatios = new std::vector<double>(size);

  for(int i = 0; i < size; ++i){
    (*oddsRatios)[i] = exp(betaCoefficents(i + 1));
  }

  return oddsRatios;
}

std::vector<double>* InteractionStatistics::calculateOddsRatiosLow(const Container::HostVector& betaCoefficents,
    const std::vector<double>& standardError) const {

  const int size = betaCoefficents.getNumberOfRows() - 1; //Skipping the intercept
  std::vector<double>* oddsRatiosLow = new std::vector<double>(size);

  for(int i = 0; i < size; ++i){
    (*oddsRatiosLow)[i] = exp(-1.96 * standardError[i + 1] + betaCoefficents(i + 1));
  }

  return oddsRatiosLow;
}

std::vector<double>* InteractionStatistics::calculateOddsRatiosHigh(const Container::HostVector& betaCoefficents,
    const std::vector<double>& standardError) const {

  const int size = betaCoefficents.getNumberOfRows() - 1; //Skipping the intercept
  std::vector<double>* oddsRatiosHigh = new std::vector<double>(size);

  for(int i = 0; i < size; ++i){
    (*oddsRatiosHigh)[i] = exp(1.96 * standardError[i + 1] + betaCoefficents(i + 1));
  }

  return oddsRatiosHigh;
}

void InteractionStatistics::toOstream(std::ostream& os) const {
  //Print AP
  os << ap << ",";

  //Print RERIR
  os << reri << ",";

  //Print ORs including covariates if any
  const int size = betaCoefficents.getNumberOfRows() - 1; //Skipping the intercept
  for(int i = 0; i < size - 1; ++i){
    os << (*(oddsRatios))[i] << "," << (*(oddsRatiosLow))[i] << "," << (*(oddsRatiosHigh))[i] << ",";
  }
  os << (*(oddsRatios))[size - 1] << "," << (*(oddsRatiosLow))[size - 1] << "," << (*(oddsRatiosHigh))[size - 1];
}

std::ostream & operator<<(std::ostream& os, const InteractionStatistics& statistics) {
  statistics.toOstream(os);
  return os;
}

} /* namespace CuEira */
