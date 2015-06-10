#include "Model.h"

namespace CuEira {
namespace Model {

Model::Model(ModelConfiguration* modelConfiguration) :
    modelConfiguration(modelConfiguration){

}

Model::~Model(){
  delete modelConfiguration;
}

} /* namespace Model */
} /* namespace CuEira */
