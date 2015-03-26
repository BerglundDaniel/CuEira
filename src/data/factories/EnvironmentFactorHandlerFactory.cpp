#include "EnvironmentFactorHandlerFactory.h"

namespace CuEira {

EnvironmentFactorHandlerFactory::EnvironmentFactorHandlerFactory() {

}

EnvironmentFactorHandlerFactory::~EnvironmentFactorHandlerFactory() {

}

EnvironmentFactorHandler* EnvironmentFactorHandlerFactory::constructEnvironmentFactorHandler(
    const Container::HostVector* envData, EnvironmentFactor* environmentFactor) const {

  bool binary = true;
  int max = (*envData)(0);
  int min = (*envData)(0);

  const int numberOfIndividuals = envData->getNumberOfRows();
  for(int i = 0; i < numberOfIndividuals; ++i){
    if((*envData)(i) != 0 && (*envData)(i) != 1){
      binary = false;
    }

    if((*envData)(i) > max){
      max = (*envData)(i);
    }

    if((*envData)(i) < min){
      min = (*envData)(i);
    }
  }

  environmentFactor->setMax(max);
  environmentFactor->setMin(min);

  if(binary){
    environmentFactor->setVariableType(BINARY);
  }else{
    environmentFactor->setVariableType(OTHER);
  }

#ifdef CPU
  return new CpuEnvironmentFactorHandler(envData, environmentFactor);
#else
  //TODO transfer to GPU
  Container::DeviceVector* envDataDevice;

  delete dataMatrix;
  return new CudaEnvironmentFactorHandler(envDataDevice, environmentFactor);
#endif
}

} /* namespace CuEira */
