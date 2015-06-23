#include "DataHandlerFactory.h"

namespace CuEira {

template<typename Matrix, typename Vector>
DataHandlerFactory<Matrix, Vector>::DataHandlerFactory(const Configuration& configuration,
    const RiskAlleleStrategy& riskAlleleStrategy, Task::DataQueue& dataQueue) :
    configuration(configuration), riskAlleleStrategy(riskAlleleStrategy), dataQueue(dataQueue){

}

template<typename Matrix, typename Vector>
DataHandlerFactory<Matrix, Vector>::~DataHandlerFactory(){

}

} /* namespace CuEira */
