#include "CpuEnvironmentFactorHandlerFactory.h"

namespace CuEira {
namespace CPU {

CpuEnvironmentFactorHandlerFactory::CpuEnvironmentFactorHandlerFactory(const Configuration& configuration,
    const std::vector<std::string>& columnNames, const Container::HostMatrix& matrix) :
    EnvironmentFactorHandlerFactory(configuration, columnNames, matrix){

}

CpuEnvironmentFactorHandlerFactory::~CpuEnvironmentFactorHandlerFactory(){

}

EnvironmentFactorHandler<Container::HostVector>* CpuEnvironmentFactorHandlerFactory::constructEnvironmentFactorHandler(
    const MKLWrapper& mklWrapper) const{
  Container::RegularHostVector* envDataTo = new Container::RegularHostVector(envData->getNumberOfRows());
  mklWrapper.copyVector(*envData, *envDataTo);

  return new EnvironmentFactorHandler<Container::HostVector>(environmentFactor, envDataTo);
}

} /* namespace CPU */
} /* namespace CuEira */
