#include "Statistics.h"

namespace CuEira {

Statistics::Statistics(Container::HostVector* betaCoefficents, Container::HostVector* standardError) :
    betaCoefficents(betaCoefficents), standardError(standardError), ap(calculateAp(reri, (*betaCoefficents)(3))), reri(
        calculateReri(oddsRatios)), oddsRatios(calculateOddsRatios(*betaCoefficents)), oddsRatiosLow(
        calculateOddsRatiosLow(*betaCoefficents, *standardError)), oddsRatiosHigh(
        calculateOddsRatiosHigh(*betaCoefficents, *standardError)) {

}

Statistics::~Statistics() {
  delete betaCoefficents;
  delete standardError;
}

double Statistics::getReri() const {
  return reri;
}

double Statistics::getAp() const {
  return ap;
}

const std::vector<double>& Statistics::getOddsRatios() const {
  return oddsRatios;
}

const std::vector<double>& Statistics::getOddsRatiosLow() const {
  return oddsRatiosLow;
}

const std::vector<double>& Statistics::getOddsRatiosHigh() const {
  return oddsRatiosHigh;
}

double Statistics::calculateReri(const std::vector<double>& oddsRatios) const {
  return oddsRatios[2] - oddsRatios[0] - oddsRatios[1] + 1;
}

double Statistics::calculateAp(double reri, PRECISION interactionBeta) const {
  return reri / interactionBeta;
}

std::vector<double> Statistics::calculateOddsRatios(const Container::HostVector& betaCoefficents) const {
  const int size = betaCoefficents.getNumberOfRows() - 1; //Skipping the intercept
  std::vector<double> oddsRatios(size);

  for(int i = 0; i < size; ++i){
    oddsRatios[i] = exp(betaCoefficents(i + 1));
  }

  return oddsRatios;
}

std::vector<double> Statistics::calculateOddsRatiosLow(const Container::HostVector& betaCoefficents,
    const Container::HostVector& standardError) const {

  const int size = betaCoefficents.getNumberOfRows() - 1; //Skipping the intercept
  std::vector<double> oddsRatiosLow(size);

  for(int i = 0; i < size; ++i){
    oddsRatiosLow[i] = exp(-1.96 * standardError(i + 1) + betaCoefficents(i + 1));
  }

  return oddsRatiosLow;
}

std::vector<double> Statistics::calculateOddsRatiosHigh(const Container::HostVector& betaCoefficents,
    const Container::HostVector& standardError) const {

  const int size = betaCoefficents.getNumberOfRows() - 1; //Skipping the intercept
  std::vector<double> oddsRatiosHigh(size);

  for(int i = 0; i < size; ++i){
    oddsRatiosHigh[i] = exp(1.96 * standardError(i + 1) + betaCoefficents(i + 1));
  }

  return oddsRatiosHigh;
}

} /* namespace CuEira */
