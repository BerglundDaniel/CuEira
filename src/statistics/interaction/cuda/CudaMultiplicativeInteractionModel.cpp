#include "CudaMultiplicativeInteractionModel.h"

namespace CuEira {
namespace CUDA {

CudaMultiplicativeInteractionModel::CudaMultiplicativeInteractionModel(const Stream& stream) :
    MultiplicativeInteractionModel<DeviceVector>(), stream(stream){

}

CudaMultiplicativeInteractionModel::~CudaMultiplicativeInteractionModel(){

}

void CudaMultiplicativeInteractionModel::applyModel(SNPVector<DeviceVector>& snpVector,
    EnvironmentVector<DeviceVector>& environmentVector, InteractionVector<DeviceVector>& interactionVector){

  interactionVector.updateSize(environmentVector.getNumberOfIndividualsToInclude());
  Kernel::elementWiseMultiplication(stream, snpVector.getSNPData(), environmentVector.getEnvironmentData(),
      interactionVector.getInteractionData());
}

} /* namespace CUDA */
} /* namespace CuEira */
