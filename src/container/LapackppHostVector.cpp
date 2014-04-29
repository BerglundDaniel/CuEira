#include "LapackppHostVector.h"

namespace CuEira {
namespace Container {

LapackppHostVector::LapackppHostVector(LaVectorDouble lapackppContainer) :
    HostVector(lapackppContainer.rows(), lapackppContainer.cols(), false, lapackppContainer.addr()), laVector(laVector) {

  if(numberOfRows != 1 && numberOfColumns != 1){
    throw DimensionMismatch("At least one dimension must be 1 to construct a vector.");
  }

}

LapackppHostVector::~LapackppHostVector() {

}

LaVectorDouble& LapackppHostVector::getLapackpp() {
  return lapackppContainer;
}

virtual double& LapackppHostVector::operator()(int index) {
  if(numberOfRows == 1){
    return lapackppContainer(1, index);
  }else{
    return lapackppContainer(index, 1);
  }
}

virtual const double& LapackppHostVector::operator()(int index) const {
  if(numberOfRows == 1){
    return lapackppContainer(1, index);
  }else{
    return lapackppContainer(index, 1);
  }
}

} /* namespace Container */
} /* namespace CuEira */
