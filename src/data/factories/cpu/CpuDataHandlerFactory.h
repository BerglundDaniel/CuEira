#ifndef CPUDATAHANDLERFACTORY_H_
#define CPUDATAHANDLERFACTORY_H_

#include <DataHandlerFactory.h>
#include <RegularHostVector.h>
#include <RegularHostMatrix.h>
#include <CpuMissingDataHandler.h>
#include <EnvironmentFactorHandler.h>
#include <EnvironmentVector.h>
#include <CpuEnvironmentVector.h>
#include <InteractionVector.h>
#include <PhenotypeVector.h>
#include <PhenotypeHandler.h>
#include <CovariatesMatrix.h>
#include <CovariatesHandler.h>
#include <ContingencyTableFactory.h>
#include <CpuContingencyTableFactory.h>
#include <RiskAlleleStrategy.h>
#include <AlleleStatisticsFactory.h>
#include <CpuAlleleStatisticsFactory.h>
#include <BedReader.h>

namespace CuEira {
namespace CPU {

using namespace CuEira::Container;

/*
 * This class....
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuDataHandlerFactory: public DataHandlerFactory<RegularHostMatrix, RegularHostVector> {
public:
  explicit CpuDataHandlerFactory(const Configuration& configuration, const RiskAlleleStrategy& riskAlleleStrategy,
      Task::DataQueue& dataQueue);
  virtual ~CpuDataHandlerFactory();

  virtual DataHandler<RegularHostMatrix, RegularHostVector>* constructDataHandler(FileIO::BedReader<>* bedReader,
      const EnvironmentFactorHandler<RegularHostVector>& environmentFactorHandler,
      const PhenotypeHandler<RegularHostVector>& phenotypeHandler,
      const CovariatesHandler<RegularHostMatrix>& covariatesHandler) const;

  CpuDataHandlerFactory(const CpuDataHandlerFactory&) = delete;
  CpuDataHandlerFactory(CpuDataHandlerFactory&&) = delete;
  CpuDataHandlerFactory& operator=(const CpuDataHandlerFactory&) = delete;
  CpuDataHandlerFactory& operator=(CpuDataHandlerFactory&&) = delete;

private:

};

} /* namespace CPU */
} /* namespace CuEira */

#endif /* CPUDATAHANDLERFACTORY_H_ */
