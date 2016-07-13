#include "DataLocation.h"

namespace CuEira {

template<typename Matrix, typename Vector>
DataLocation<Matrix, Vector>::DataLocation(Vector& snpVector, Vector& environmentVector, Vector& interactionVector,
    Vector& phenotypeVector, Matrix& covariatesMatrix) :
    snpVector(SNPVector), environmentVector(environmentVector), interactionVector(interactionVector), phenotypeVector(
        phenotypeVector), covariatesMatrix(covariatesMatrix){
//TODO error check lengths
}

template<typename Matrix, typename Vector>
DataLocation<Matrix, Vector>::~DataLocation(){

}

} /* namespace CuEira */
