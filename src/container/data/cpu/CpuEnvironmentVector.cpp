#include "CpuEnvironmentVector.h"

namespace CuEira {
namespace Container {
namespace CPU {

CpuEnvironmentVector::CpuEnvironmentVector(const CpuEnvironmentFactorHandler& cpuEnvironmentFactorHandler,
    const MKLWrapper& mklWrapper) :
    EnvironmentVector(cpuEnvironmentFactorHandler.getEnvironmentFactor(),
        cpuEnvironmentFactorHandler.getNumberOfIndividualsTotal()), originalData(
        cpuEnvironmentFactorHandler.getEnvironmentData()), recodedData(nullptr), mklWrapper(mklWrapper) {

}

CpuEnvironmentVector::~CpuEnvironmentVector() {
  delete recodedData;
}

const Container::HostVector& CpuEnvironmentVector::getEnvironmentData() const {
#ifdef DEBUG
  if(!initialised){
    throw InvalidState("CudaEnvironmentVector not initialised.");
  }
#endif

  return *recodedData;
}

void CpuEnvironmentVector::recode(Recode recode) {
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
    mklWrapper.copyVector(originalData, *recodedData);
  }

  noMissing = true;
}

void CpuEnvironmentVector::recode(Recode recode, const CpuMissingDataHandler& missingDataHandler) {
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

void CpuEnvironmentVector::recodeProtective() {
  int c;
  if(environmentFactor.getVariableType() == BINARY){
    c = 1;
  }else{
    c = environmentFactor.getMax() + environmentFactor.getMin();
  }

  //UNROLL or maybe MKL?
  for(int i = 0; i < numberOfIndividualsToInclude; ++i){
    (*recodedData)(i) = c - (*recodedData)(i);
  }
}

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */
