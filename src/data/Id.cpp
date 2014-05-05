#include "Id.h"

namespace CuEira {

Id::Id(std::string id) :
    id(id) {

}

Id::~Id() {

}

const std::string Id::getString() const {
  return id;
}

bool Id::operator<(const Id& otherId) const {
  if(id < otherId.getString()){
    return true;
  }else{
    return false;
  }
}

bool Id::operator==(const Id& otherId) const {
  if(id == otherId.getString()){
    return true;
  }else{
    return false;
  }
}

} /* namespace CuEira */
