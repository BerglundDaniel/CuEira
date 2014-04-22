#include "Id.h"

namespace CuEira {

Id::Id(std::string id) :
    id(id) {

}

Id::~Id() {

}

const std::string Id::getString(){
  return id;
}

} /* namespace CuEira */
