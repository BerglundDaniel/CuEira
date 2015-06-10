#include "CudaAdditiveInteractionModel.h"

namespace CuEira {
namespace CUDA {

CudaAdditiveInteractionModel::CudaAdditiveInteractionModel(const Stream& stream) :
    AdditiveInteractionModel<DeviceVector>(), stream(stream){

}

CudaAdditiveInteractionModel::~CudaAdditiveInteractionModel(){

}

void CudaAdditiveInteractionModel::applyModel(SNPVector<DeviceVector>& snpVector,
    EnvironmentVector<DeviceVector>& environmentVector, InteractionVector<DeviceVector>& interactionVector){
  interactionVector.updateSize(environmentVector.getNumberOfIndividualsToInclude());
  Kernel::applyAdditiveModel(stream, snpVector.getSNPData(), environmentVector.getEnvironmentData(),
      interactionVector.getInteractionData());
}

} /* namespace CUDA */
} /* namespace CuEira */
