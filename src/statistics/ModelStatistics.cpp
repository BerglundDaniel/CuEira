#include "ModelStatistics.h"

namespace CuEira {

ModelStatistics::ModelStatistics() {

}

ModelStatistics::~ModelStatistics() {

}

std::ostream & operator<<(std::ostream& os, const ModelStatistics& modelStatistics) {
  modelStatistics.toOstream(os);
  return os;
}

} /* namespace CuEira */
