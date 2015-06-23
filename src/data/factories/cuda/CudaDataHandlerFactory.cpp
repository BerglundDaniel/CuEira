#include "CudaDataHandlerFactory.h"

namespace CuEira {
namespace CUDA {

CudaDataHandlerFactory::CudaDataHandlerFactory(const Configuration& configuration,
    const RiskAlleleStrategy& riskAlleleStrategy, Task::DataQueue& dataQueue) :
    DataHandlerFactory(configuration, riskAlleleStrategy, dataQueue){

}

CudaDataHandlerFactory::~CudaDataHandlerFactory(){

}

DataHandler<DeviceMatrix, DeviceVector>* CudaDataHandlerFactory::constructDataHandler() const{
  //TODO

  return new DataHandler<DeviceMatrix, DeviceVector>();
}

} /* namespace CUDA */
} /* namespace CuEira */
