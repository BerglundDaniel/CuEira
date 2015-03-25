#include "CpuEnvironmentFactorHandler.h"

namespace CuEira {
namespace CPU {

CpuEnvironmentFactorHandler::CpuEnvironmentFactorHandler(const Container::HostMatrix* dataMatrix,
    const std::vector<const EnvironmentFactor*>* environmentFactors, const std::vector<std::set<int>>* personsToSkip) :
    EnvironmentFactorHandler(environmentFactors, personsToSkip), dataMatrix(dataMatrix) {

}

CpuEnvironmentFactorHandler::~CpuEnvironmentFactorHandler() {
  delete dataMatrix;
}

Container::CPU::CpuEnvironmentVector* CpuEnvironmentFactorHandler::getEnvironmentVector(
    const EnvironmentFactor& environmentFactor) const {
  for(int i = 0; i < numberOfColumns; ++i){
    if(*(*environmentFactors)[i] == environmentFactor){
      const Container::HostVector* vector = (*dataMatrix)(i);
      return vector; //TODO create a env vector with stuff
    } // if
  } // for i

  std::ostringstream os;
  os << "Can't find EnvironmentFactor " << environmentFactor.getId().getString() << " in EnvironmentFactorHandler."
      << std::endl;
  const std::string& tmp = os.str();
  throw EnvironmentFactorHandlerException(tmp.c_str());
}

} /* namespace CPU */
} /* namespace CuEira */
