#include "CombinedResults.h"

namespace CuEira {
namespace Model {

CombinedResults::CombinedResults(InteractionStatistics* interactionStatistics, Recode recode) :
    interactionStatistics(interactionStatistics), recode(recode) {

}

CombinedResults::~CombinedResults() {
  delete interactionStatistics;
}

void CombinedResults::toOstream(std::ostream& os) const {
  os << *interactionStatistics << "," << recode;
}

std::ostream & operator<<(std::ostream& os, const CombinedResults& combinedResults) {
  combinedResults.toOstream(os);
  return os;
}

} /* namespace Model */
} /* namespace CuEira */
