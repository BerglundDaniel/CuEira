#include "LapackppHostVector.h"

namespace CuEira {
namespace Container {

LapackppHostVector::LapackppHostVector(int numberOfIndividualsToInclude) : lapackppContainer(numberOfIndividualsToInclude),
    HostVector(lapackppContainer.rows(), lapackppContainer.cols(), false, lapackppContainer.addr()) {
}

LapackppHostVector::~LapackppHostVector() {

}

LaVectorDouble& LapackppHostVector::getLapackpp() {
  return lapackppContainer;
}

double& LapackppHostVector::operator()(int index) {
  return lapackppContainer(index);
}

const double& LapackppHostVector::operator()(int index) const {
  return lapackppContainer(index);
}

} /* namespace Container */
} /* namespace CuEira */
