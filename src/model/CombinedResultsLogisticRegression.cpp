#include "CombinedResultsLogisticRegression.h"

namespace CuEira {
namespace Model {

CombinedResultsLogisticRegression::CombinedResultsLogisticRegression(InteractionStatistics* additiveStatistics,
    OddsRatioStatistics* multiplicativeStatistics, Recode recode) :
    additiveStatistics(additiveStatistics), multiplicativeStatistics(multiplicativeStatistics), recode(recode) {

}

CombinedResultsLogisticRegression::CombinedResultsLogisticRegression() :
    additiveStatistics(nullptr), multiplicativeStatistics(nullptr), recode(ALL_RISK) {

}

CombinedResultsLogisticRegression::~CombinedResultsLogisticRegression() {
  delete additiveStatistics;
  delete multiplicativeStatistics;
}

void CombinedResultsLogisticRegression::toOstream(std::ostream& os) const {
  os << *additiveStatistics << "," << *multiplicativeStatistics << "," << recode;
}

std::ostream& operator<<(std::ostream& os, const CombinedResultsLogisticRegression& combinedResultsLogisticRegression) {
  combinedResultsLogisticRegression.toOstream(os);
  return os;
}

} /* namespace Model */
} /* namespace CuEira */
