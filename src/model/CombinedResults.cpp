#include "CombinedResults.h"

namespace CuEira {
namespace Model {

CombinedResults::CombinedResults() {

}

CombinedResults::~CombinedResults() {

}

std::ostream & operator<<(std::ostream& os, const CombinedResults& combinedResults) {
  combinedResults.toOstream(os);
  return os;
}

} /* namespace Model */
} /* namespace CuEira */
