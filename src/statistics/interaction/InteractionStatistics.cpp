#include "InteractionStatistics.h"

namespace CuEira {

InteractionStatistics::InteractionStatistics(
    const Model::LogisticRegression::LogisticRegressionResult* logisticRegressionResult) :
    OddsRatioStatistics(logisticRegressionResult), ap(calculateAp(reri, (*betaCoefficents)(3))), reri(
        calculateReri(*oddsRatios)) {

}

InteractionStatistics::InteractionStatistics() :
    OddsRatioStatistics(), ap(0), reri(0) {

}

InteractionStatistics::~InteractionStatistics() {

}

double InteractionStatistics::getReri() const {
  return reri;
}

double InteractionStatistics::getAp() const {
  return ap;
}

double InteractionStatistics::calculateReri(const std::vector<double>& oddsRatios) const {
  return oddsRatios[2] - oddsRatios[0] - oddsRatios[1] + 1;
}

double InteractionStatistics::calculateAp(double reri, PRECISION interactionBeta) const {
  return reri / interactionBeta;
}

void InteractionStatistics::toOstream(std::ostream& os) const {
  //Print AP
  os << ap << ",";

  //Print RERIR
  os << reri << ",";

  OddsRatioStatistics::toOstream(os);
}

std::ostream & operator<<(std::ostream& os, const InteractionStatistics& statistics) {
  statistics.toOstream(os);
  return os;
}

} /* namespace CuEira */
