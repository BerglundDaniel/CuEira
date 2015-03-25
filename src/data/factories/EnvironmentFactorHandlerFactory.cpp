#include "EnvironmentFactorHandlerFactory.h"

namespace CuEira {

EnvironmentFactorHandlerFactory::EnvironmentFactorHandlerFactory() {

}

EnvironmentFactorHandlerFactory::~EnvironmentFactorHandlerFactory() {

}

EnvironmentFactorHandler* EnvironmentFactorHandlerFactory::constructEnvironmentFactorHandler(
    const Container::HostMatrix* dataMatrix, const std::vector<EnvironmentFactor*>* environmentFactors) const {
  if(dataMatrix->getNumberOfColumns() != environmentFactors->size()){
    std::ostringstream os;
    os << "Number of columns in dataMatrix and number of environmental factors does not match." << std::endl;
    const std::string& tmp = os.str();
    throw EnvironmentFactorHandlerException(tmp.c_str());
  }

  const int numberOfEnvironmentFactors = environmentFactors->size();
  std::vector<const EnvironmentFactor*>* constEnvironmentFactors = new std::vector<const EnvironmentFactor*>(
      numberOfEnvironmentFactors);

  //Check the variable type for each factor
  //TODO potential unroll
  for(int i = 0; i < numberOfEnvironmentFactors; ++i){
    bool binary = true;
    EnvironmentFactor* environmentFactor = (*environmentFactors)[i];

    constEnvironmentFactors[i] = environmentFactor; //Set the factor in the constFactor vector

    //TODO potential unroll
    for(int j = 0; j < numberOfIndividualsToInclude; ++j){
      if((*dataMatrix)(j, i) != 0 && (*dataMatrix)(j, i) != 1){
        binary = false;
        break;
      }
    } //for j

    if(binary){
      environmentFactor->setVariableType(BINARY);
    }else{
      environmentFactor->setVariableType(OTHER);
    }
  } // for i

  delete environmentFactors; //Since they are in the const vector instead

#ifdef CPU
  return new CpuEnvironmentFactorHandler(dataMatrix, constEnvironmentFactors, personsToSkip);
#else
  //TODO transfer to GPU
  Container::DeviceMatrix* dataMatrixDevice;

  delete dataMatrix;
  return new CudaEnvironmentFactorHandler(dataMatrixDevice, constEnvironmentFactors, personsToSkip);
#endif
}

} /* namespace CuEira */
