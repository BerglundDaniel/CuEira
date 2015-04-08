#include "CudaAdditiveInteractionModel.h"

namespace CuEira {
namespace CUDA {

CudaAdditiveInteractionModel::CudaAdditiveInteractionModel(const KernelWrapper& kernelWrapper) :
    AdditiveInteractionModel<DeviceVector>(), kernelWrapper(kernelWrapper) {

}

CudaAdditiveInteractionModel::~CudaAdditiveInteractionModel() {

}

void CudaAdditiveInteractionModel::applyModel(SNPVector<DeviceVector>& snpVector,
    EnvironmentVector<DeviceVector>& environmentVector, InteractionVector<DeviceVector>& interactionVector) {
  interactionVector.updateSize(environmentVector.getNumberOfIndividualsToInclude());
  kernelWrapper.applyAdditiveModel(snpVector.getSNPData(), environmentVector.getEnvironmentData(),
      interactionVector.getInteractionData());
}

} /* namespace CUDA */
} /* namespace CuEira */
