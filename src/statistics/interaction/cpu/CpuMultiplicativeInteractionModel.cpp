#include "CpuMultiplicativeInteractionModel.h"

namespace CuEira {
namespace CPU {

CpuMultiplicativeInteractionModel::CpuMultiplicativeInteractionModel(const MKLWrapper& mklWrapper) :
    MultiplicativeInteractionModel<RegularHostVector>(), mklWrapper(mklWrapper) {

}

CpuMultiplicativeInteractionModel::~CpuMultiplicativeInteractionModel() {

}

void CpuMultiplicativeInteractionModel::applyModel(SNPVector<RegularHostVector>& snpVector,
    EnvironmentVector<RegularHostVector>& environmentVector, InteractionVector<RegularHostVector>& interactionVector) {
  interactionVector.updateSize(environmentVector.getNumberOfIndividualsToInclude());
  mklWrapper.multiplicationElementWise(snpVector.getSNPData(), environmentVector.getEnvironmentData(),
      interactionVector.getInteractionData());
}

} /* namespace CPU */
} /* namespace CuEira */
