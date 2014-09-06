#include "DataHandlerFactory.h"

namespace CuEira {

DataHandlerFactory::DataHandlerFactory(const Configuration& configuration,
    const ContingencyTableFactory& contingencyTableFactory) :
    configuration(configuration), contingencyTableFactory(contingencyTableFactory) {

}

DataHandlerFactory::~DataHandlerFactory() {

}

DataHandler* DataHandlerFactory::constructDataHandler(const FileIO::BedReader& bedReader,
    const EnvironmentFactorHandler& environmentFactorHandler, Task::DataQueue& dataQueue) const {
  Container::EnvironmentVector* environmentVector = new Container::EnvironmentVector(*environmentFactorHandler);
  Container::InteractionVector* interactionVector = new Container::InteractionVector(*environmentVector);

  return new DataHandler(configuration, bedReader, contingencyTableFactory, environmentFactorHandler.getHeaders(),
      dataQueue, environmentVector, interactionVector);
}

} /* namespace CuEira */
