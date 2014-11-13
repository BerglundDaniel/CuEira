#include "LogisticRegressionCombinedResults.h"

namespace CuEira {
namespace Model {
namespace LogisticRegression {

LogisticRegressionCombinedResults::LogisticRegressionCombinedResults(InteractionStatistics* additiveStatistics,
    OddsRatioStatistics* multiplicativeStatistics, Recode recode) :
    additiveStatistics(additiveStatistics), multiplicativeStatistics(multiplicativeStatistics), recode(recode) {

}

LogisticRegressionCombinedResults::LogisticRegressionCombinedResults() :
    additiveStatistics(nullptr), multiplicativeStatistics(nullptr), recode(ALL_RISK) {

}

LogisticRegressionCombinedResults::~LogisticRegressionCombinedResults() {
  delete additiveStatistics;
  delete multiplicativeStatistics;
}

void LogisticRegressionCombinedResults::toOstream(std::ostream& os) const {
  os << *additiveStatistics << "," << *multiplicativeStatistics << "," << recode;
}

std::ostream& operator<<(std::ostream& os, const LogisticRegressionCombinedResults& logisticRegressionCombinedResults) {
  logisticRegressionCombinedResults.toOstream(os);
  return os;
}

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
