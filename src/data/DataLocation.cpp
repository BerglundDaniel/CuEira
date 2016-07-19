#include "DataLocation.h"

namespace CuEira {

template<typename Matrix, typename Vector>
DataLocation<Matrix, Vector>::DataLocation(Vector* snpVector, Vector* environmentVector, Vector* interactionVector,
    Vector* phenotypeVector, Matrix* covariatesMatrix) :
    snpVector(snpVector), environmentVector(environmentVector), interactionVector(interactionVector), phenotypeVector(
        phenotypeVector), covariatesMatrix(covariatesMatrix){
#ifdef DEBUG
  int sLength = snpVector.getRealNumberOfRows();
  int eLength = environmentVector.getRealNumberOfRows();
  int iLength = interactionVector.getRealNumberOfRows();
  int pLength = phenotypeVector.getRealNumberOfRows();
  int cLength = covariatesMatrix.getRealNumberOfRows();

  if(sLength != eLength || sLength != iLength || sLength != pLength || sLength != cLength){
    std::ostringstream os;
    os << "DataLocation number of rows of containers does not match" << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }


#endif
}

template<typename Matrix, typename Vector>
DataLocation<Matrix, Vector>::~DataLocation(){
  //All or some of these containers can be subviews so DataLocation might not own the memory area of the containers
  delete snpVector;
  delete environmentVector;
  delete interactionVector;
  delete phenotypeVector;
  delete covariatesMatrix;
}

template<typename Matrix, typename Vector>
Vector& DataLocation<Matrix, Vector>::getSnpVector(){
  return snpVector;
}

template<typename Matrix, typename Vector>
Vector& DataLocation<Matrix, Vector>::getEnvironmentVector(){
  return environmentVector;
}

template<typename Matrix, typename Vector>
Vector& DataLocation<Matrix, Vector>::getInteractionVector(){
  return interactionVector;
}

template<typename Matrix, typename Vector>
Vector& DataLocation<Matrix, Vector>::getPhenotypeVector(){
  return phenotypeVector;
}

template<typename Matrix, typename Vector>
Matrix& DataLocation<Matrix, Vector>::getCovariatesMatrix(){
  return covariatesMatrix;
}

} /* namespace CuEira */
