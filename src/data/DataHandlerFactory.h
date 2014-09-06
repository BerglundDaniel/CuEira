#ifndef DATAHANDLERFACTORY_H_
#define DATAHANDLERFACTORY_H_

#include <Configuration.h>
#include <DataHandler.h>
#include <BedReader.h>
#include <DataQueue.h>
#include <Configuration.h>
#include <ContingencyTableFactory.h>
#include <EnvironmentFactor.h>
#include <EnvironmentFactorHandler.h>
#include <EnvironmentVector.h>
#include <InteractionVector.h>

namespace CuEira {

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataHandlerFactory {
public:
  DataHandlerFactory(const Configuration& configuration, const ContingencyTableFactory& contingencyTableFactory);
  virtual ~DataHandlerFactory();

  virtual DataHandler* constructDataHandler(const FileIO::BedReader& bedReader,
      const EnvironmentFactorHandler& environmentFactorHandler, Task::DataQueue& dataQueue) const;

private:
  const Configuration& configuration;
  const ContingencyTableFactory& contingencyTableFactory;
};

} /* namespace CuEira */

#endif /* DATAHANDLERFACTORY_H_ */
