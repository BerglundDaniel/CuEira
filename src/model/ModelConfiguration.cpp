#include "ModelConfiguration.h"

namespace CuEira {
namespace Model {

ModelConfiguration::ModelConfiguration(const Configuration& configuration, const MKLWrapper& blasWrapper) :
    configuration(configuration), blasWrapper(blasWrapper) {

}

ModelConfiguration::~ModelConfiguration() {

}

const MKLWrapper& ModelConfiguration::getBlasWrapper() const {
  return blasWrapper;
}

} /* namespace Model */
} /* namespace CuEira */
