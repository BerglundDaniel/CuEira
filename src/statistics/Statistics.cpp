#include "Statistics.h"

namespace CuEira {

Statistics::Statistics(const Container::DeviceVector& betaCoefficents) :
    ap(calculateAp(reri, betaCoefficents(3))), rerri(calculateReri(oddsRatios)), oddsRatios(
        calculateOddsRatios(betaCoefficents)), oddsRatiosLow(calculateOddsRatiosLow(betaCoefficents)), oddsRatiosHigh(
        calculateOddsRatiosHigh(betaCoefficents)) {

}

Statistics::~Statistics() {

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

std::vector<double> Statistics::calculateOddsRatios(const Container::DeviceVector& betaCoefficents) const {
  const int size = betaCoefficents.getNumberOfRows() - 1; //Skipping the intercept
  std::vector<double> oddsRatios(size);

  for(int i = 0; i < size; ++i){
    oddsRatios[i] = betaCoefficents(i + 1);
  }

  return oddsRatios;
}

std::vector<double> Statistics::calculateOddsRatiosLow(const Container::DeviceVector& betaCoefficents) const {

}

std::vector<double> Statistics::calculateOddsRatiosHigh(const Container::DeviceVector& betaCoefficents) const {

}

} /* namespace CuEira */
