#include "CudaMultiplicativeInteractionModel.h"

namespace CuEira {
namespace CUDA {

CudaMultiplicativeInteractionModel::CudaMultiplicativeInteractionModel(const KernelWrapper& kernelWrapper) :
    MultiplicativeInteractionModel<DeviceVector>(), kernelWrapper(kernelWrapper) {

}

CudaMultiplicativeInteractionModel::~CudaMultiplicativeInteractionModel() {

}

void CudaMultiplicativeInteractionModel::applyModel(SNPVector<DeviceVector>& snpVector,
    EnvironmentVector<DeviceVector>& environmentVector, InteractionVector<DeviceVector>& interactionVector) {

  interactionVector.updateSize(environmentVector.getNumberOfIndividualsToInclude());
  kernelWrapper.elementWiseMultiplication(snpVector.getSNPData(), environmentVector.getEnvironmentData(),
      interactionVector.getInteractionData());
}

} /* namespace CUDA */
} /* namespace CuEira */
