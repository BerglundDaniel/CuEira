#include "ModelInformation.h"

namespace CuEira {
namespace Model {

ModelInformation::ModelInformation(ModelState modelState, std::string information) :
    modelState(modelState), information(information) {

}

ModelInformation::~ModelInformation() {

}

ModelState ModelInformation::getModelState() const {
  return modelState;
}

void ModelInformation::toOstream(std::ostream& os) const {
  os << information;
}

std::ostream & operator<<(std::ostream& os, const ModelInformation& modelInformation) {
  modelInformation.toOstream(os);
  return os;
}

} /* namespace Model */
} /* namespace CuEira */
