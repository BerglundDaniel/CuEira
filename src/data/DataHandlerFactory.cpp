#include "DataHandlerFactory.h"

namespace CuEira {

DataHandlerFactory::DataHandlerFactory(const Configuration& configuration,
    const ContingencyTableFactory& contingencyTableFactory, FileIO::BedReader& bedReader,
    const EnvironmentFactorHandler& environmentFactorHandler, Task::DataQueue& dataQueue) :
    configuration(configuration), contingencyTableFactory(contingencyTableFactory), bedReader(bedReader), environmentFactorHandler(
        environmentFactorHandler), dataQueue(dataQueue) {

}

DataHandlerFactory::~DataHandlerFactory() {

}

DataHandler* DataHandlerFactory::constructDataHandler() const {
  Container::EnvironmentVector* environmentVector = new Container::EnvironmentVector(environmentFactorHandler);
  Container::InteractionVector* interactionVector = new Container::InteractionVector(*environmentVector);

  return new DataHandler(configuration, bedReader, contingencyTableFactory, environmentFactorHandler.getHeaders(),
      dataQueue, environmentVector, interactionVector);
}

} /* namespace CuEira */
