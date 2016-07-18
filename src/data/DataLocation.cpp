#include "DataLocation.h"

namespace CuEira {

template<typename Matrix, typename Vector>
DataLocation<Matrix, Vector>::DataLocation(Vector& snpVector, Vector& environmentVector, Vector& interactionVector,
    Vector& phenotypeVector, Matrix& covariatesMatrix) :
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
    os << "Data location number of rows does not match. " << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
#endif
}

template<typename Matrix, typename Vector>
DataLocation<Matrix, Vector>::~DataLocation(){

}

} /* namespace CuEira */
