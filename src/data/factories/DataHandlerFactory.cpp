#include "DataHandlerFactory.h"

namespace CuEira {

DataHandlerFactory::DataHandlerFactory(const Configuration& configuration, FileIO::BedReader& bedReader,
    const ContingencyTableFactory& contingencyTableFactory,
    const Model::ModelInformationFactory& modelInformationFactory,
    const EnvironmentFactorHandler& environmentFactorHandler, Task::DataQueue& dataQueue) :
    configuration(configuration), contingencyTableFactory(contingencyTableFactory), bedReader(bedReader), environmentFactorHandler(
        environmentFactorHandler), dataQueue(dataQueue), modelInformationFactory(modelInformationFactory) {

}

DataHandlerFactory::~DataHandlerFactory() {

}

DataHandler* DataHandlerFactory::constructDataHandler() const {
  Container::EnvironmentVector* environmentVector = new Container::EnvironmentVector(environmentFactorHandler);
  Container::InteractionVector* interactionVector = new Container::InteractionVector(*environmentVector);

  return new DataHandler(configuration, bedReader, contingencyTableFactory, modelInformationFactory,
      environmentFactorHandler.getHeaders(), dataQueue, environmentVector, interactionVector);
}

} /* namespace CuEira */
