#include "CudaEnvironmentFactorHandler.h"

namespace CuEira {
namespace CUDA {

CudaEnvironmentFactorHandler::CudaEnvironmentFactorHandler(const Container::DeviceMatrix* dataMatrix,
    const std::vector<const EnvironmentFactor*>* environmentFactors, const std::vector<std::set<int>>* personsToSkip) :
    EnvironmentFactorHandler(environmentFactors, personsToSkip), dataMatrix(dataMatrix) {

}

CudaEnvironmentFactorHandler::~CudaEnvironmentFactorHandler() {
  delete dataMatrix;
}

Container::CUDA::CudaEnvironmentVector* CudaEnvironmentFactorHandler::getEnvironmentVector(
    const EnvironmentFactor& environmentFactor) const {
  for(int i = 0; i < numberOfColumns; ++i){
    if(*(*environmentFactors)[i] == environmentFactor){
      const Container::DeviceVector* vector = (*dataMatrix)(i);
      return vector; //TODO create a env vector with stuff
    } // if
  } // for i

  std::ostringstream os;
  os << "Can't find EnvironmentFactor " << environmentFactor.getId().getString() << " in EnvironmentFactorHandler."
      << std::endl;
  const std::string& tmp = os.str();
  throw EnvironmentFactorHandlerException(tmp.c_str());
}

} /* namespace CUDA */
} /* namespace CuEira */
