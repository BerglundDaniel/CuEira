#include "LapackppHostMatrix.h"

namespace CuEira {
namespace Container {

LapackppHostMatrix::LapackppHostMatrix(LaGenMatDouble* lapackppContainer) :
    HostMatrix(lapackppContainer->rows(), lapackppContainer->cols(), lapackppContainer->addr()), lapackppContainer(
        lapackppContainer) {

}

LapackppHostMatrix::~LapackppHostMatrix() {
  delete lapackppContainer;
}

LaGenMatDouble& LapackppHostMatrix::getLapackpp() {
  return *lapackppContainer;
}

HostVector* LapackppHostMatrix::operator()(int column) {
  LaVectorDouble* laVector = new LaVectorDouble(numberOfRows);
  laVector->ref(lapackppContainer->col(column));
  return new LapackppHostVector(laVector, true);
}

const HostVector* LapackppHostMatrix::operator()(int column) const {
  LaVectorDouble* laVector = new LaVectorDouble(numberOfRows);
  laVector->ref(lapackppContainer->col(column));
  return new LapackppHostVector(laVector, true);
}

double& LapackppHostMatrix::operator()(int row, int column) {
  return (*lapackppContainer)(row, column);
}

const double& LapackppHostMatrix::operator()(int row, int column) const {
  return (*lapackppContainer)(row, column);
}

}
/* namespace Container */
} /* namespace CuEira */
