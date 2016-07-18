#include "CpuAdditiveInteractionModel.h"

namespace CuEira {
namespace CPU {

CpuAdditiveInteractionModel::CpuAdditiveInteractionModel() :
    AdditiveInteractionModel<RegularHostVector>(){

}

CpuAdditiveInteractionModel::~CpuAdditiveInteractionModel(){

}

void CpuAdditiveInteractionModel::applyModel(SNPVector<RegularHostVector>& snpVector,
    EnvironmentVector<RegularHostVector>& environmentVector, InteractionVector<RegularHostVector>& interactionVector){
  const int numberOfIndividualsToInclude = environmentVector.getNumberOfIndividualsToInclude();
  interactionVector.updateSize(numberOfIndividualsToInclude);

  RegularHostVector& snpData = snpVector.getSNPData();
  RegularHostVector& envData = environmentVector.getEnvironmentData();
  RegularHostVector& interactionData = interactionVector.getInteractionData();

  Blas::multiplicationElementWise(snpData, envData, interactionData);

  //UNROLL
  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    if(interactionData(i) != 0){
      snpData(i) = 0;
      envData(i) = 0;
    }
  }
}

} /* namespace CPU */
} /* namespace CuEira */
