#include "CpuMultiplicativeInteractionModel.h"

namespace CuEira {
namespace CPU {

CpuMultiplicativeInteractionModel::CpuMultiplicativeInteractionModel() :
    MultiplicativeInteractionModel<RegularHostVector>(){

}

CpuMultiplicativeInteractionModel::~CpuMultiplicativeInteractionModel(){

}

void CpuMultiplicativeInteractionModel::applyModel(SNPVector<RegularHostVector>& snpVector,
    EnvironmentVector<RegularHostVector>& environmentVector, InteractionVector<RegularHostVector>& interactionVector){
  interactionVector.updateSize(environmentVector.getNumberOfIndividualsToInclude());
  Blas::multiplicationElementWise(snpVector.getSNPData(), environmentVector.getEnvironmentData(),
      interactionVector.getInteractionData());
}

} /* namespace CPU */
} /* namespace CuEira */
