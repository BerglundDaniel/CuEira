#include "LapackppHostVector.h"

namespace CuEira {
namespace Container {

LapackppHostVector::LapackppHostVector(LaVectorDouble* lapackppContainer) :
    HostVector(lapackppContainer->size(), false, lapackppContainer->addr()), lapackppContainer(lapackppContainer) {

}

LapackppHostVector::LapackppHostVector(LaVectorDouble* lapackppContainer, bool subview) :
    HostVector(lapackppContainer->size(), subview, lapackppContainer->addr()), lapackppContainer(lapackppContainer) {

}

LapackppHostVector::~LapackppHostVector() {
   delete lapackppContainer;
}

LaVectorDouble& LapackppHostVector::getLapackpp() {
  return *lapackppContainer;
}

double& LapackppHostVector::operator()(int index) {
  return (*lapackppContainer)(index);
}

const double& LapackppHostVector::operator()(int index) const {
  return (*lapackppContainer)(index);
}

} /* namespace Container */
} /* namespace CuEira */
