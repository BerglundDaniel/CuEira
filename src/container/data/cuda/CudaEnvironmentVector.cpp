#include "CudaEnvironmentVector.h"

namespace CuEira {
namespace Container {
namespace CUDA {

CudaEnvironmentVector::CudaEnvironmentVector(const CudaEnvironmentFactorHandler& cudaEnvironmentFactorHandler,
    const KernelWrapper& kernelWrapper, const CublasWrapper& cublasWrapper) :
    EnvironmentVector(cudaEnvironmentFactorHandler.getEnvironmentFactor(),
        cudaEnvironmentFactorHandler.getNumberOfIndividualsTotal()), originalData(
        cudaEnvironmentFactorHandler.getEnvironmentData()), recodedData(nullptr), kernelWrapper(kernelWrapper), cublasWrapper(
        cublasWrapper) {

}

CudaEnvironmentVector::~CudaEnvironmentVector() {
  delete recodedData;
}

const Container::DeviceVector& CudaEnvironmentVector::getEnvironmentData() const {
#ifdef DEBUG
  if(!initialised){
    throw InvalidState("CudaEnvironmentVector not initialised.");
  }
#endif

  return *recodedData;
}

void CudaEnvironmentVector::recode(Recode recode) {
#ifdef DEBUG
  initialised=true;
#endif

  currentRecode = recode;
  if(!noMissing){
    delete recodedData;
    recodedData = new DeviceVector(numberOfIndividualsTotal);
    numberOfIndividualsToInclude = numberOfIndividualsTotal;
  }

  if(recode == ENVIRONMENT_PROTECT || recode == INTERACTION_PROTECT){
    recodeProtective();
  }else{
    cublasWrapper.copyVector(originalData, *recodedData);
  }

  noMissing = true;
}

void CudaEnvironmentVector::recode(Recode recode, const CudaMissingDataHandler& missingDataHandler) {
#ifdef DEBUG
  initialised=true;
#endif

  currentRecode = recode;
  numberOfIndividualsToInclude = missingDataHandler.getNumberOfIndividualsToInclude();
  delete recodedData;
  recodedData = missingDataHandler.copyNonMissing(originalData);

  if(recode == ENVIRONMENT_PROTECT || recode == INTERACTION_PROTECT){
    recodeProtective();
  }
}

void CudaEnvironmentVector::recodeProtective() {
  if(environmentFactor.getVariableType() == BINARY){
    kernelWrapper.constSubtractVector(1, *recodedData);
  }else{
    kernelWrapper.constSubtractVector(environmentFactor.getMax() + environmentFactor.getMin(), *recodedData);
  }
}

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */
