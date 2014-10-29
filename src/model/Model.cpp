#include "Model.h"

namespace CuEira {
namespace Model {

Model::Model(ModelConfiguration* modelConfiguration) :
    modelConfiguration(modelConfiguration), blasWrapper(&modelConfiguration->getBlasWrapper()) {

}

Model::Model() :
    modelConfiguration(nullptr), blasWrapper(nullptr) {

}

Model::~Model() {
  delete modelConfiguration;
}

} /* namespace Model */
} /* namespace CuEira */
