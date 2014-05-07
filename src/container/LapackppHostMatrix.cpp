#include "LapackppHostMatrix.h"

namespace CuEira {
namespace Container {

LapackppHostMatrix::LapackppHostMatrix(LaGenMatDouble lapackppContainer) :
    HostMatrix(lapackppContainer.rows(), lapackppContainer.cols(), lapackppContainer.addr()), lapackppContainer(
        lapackppContainer) {

}

LapackppHostMatrix::~LapackppHostMatrix() {

}

LaGenMatDouble& LapackppHostMatrix::getLapackpp() {
  return lapackppContainer;
}

HostVector* LapackppHostMatrix::operator()(int column) {
  return new LapackppHostVector(lapackppContainer.col(column), false);
}

const HostVector* LapackppHostMatrix::operator()(int column) const {
  return new LapackppHostVector(lapackppContainer.col(column), false);
}

double& LapackppHostMatrix::operator()(int row, int column) {
  return lapackppContainer(row, column);
}

const double& LapackppHostMatrix::operator()(int row, int column) const {
  return lapackppContainer(row, column);
}

}
/* namespace Container */
} /* namespace CuEira */
