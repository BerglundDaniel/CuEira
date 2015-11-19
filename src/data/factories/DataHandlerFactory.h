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
#include <ModelInformationFactory.h>

namespace CuEira {

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Matrix, typename Vector>
class DataHandlerFactory {
public:
  explicit DataHandlerFactory(const Configuration& configuration, const RiskAlleleStrategy& riskAlleleStrategy,
      Task::DataQueue& dataQueue);
  virtual ~DataHandlerFactory();

  virtual DataHandler<Matrix, Vector>* constructDataHandler(const FileIO::BedReader<Vector>* bedReader,
      const EnvironmentFactorHandler<Vector>& environmentFactorHandler,
      const PhenotypeHandler<Vector>& phenotypeHandler, const CovariatesHandler<Matrix>& covariatesHandler) const=0;

  DataHandlerFactory(const DataHandlerFactory&) = delete;
  DataHandlerFactory(DataHandlerFactory&&) = delete;
  DataHandlerFactory& operator=(const DataHandlerFactory&) = delete;
  DataHandlerFactory& operator=(DataHandlerFactory&&) = delete;

protected:
  const Configuration& configuration;
  const RiskAlleleStrategy& riskAlleleStrategy;
  Task::DataQueue& dataQueue;
};

} /* namespace CuEira */

#endif /* DATAHANDLERFACTORY_H_ */
