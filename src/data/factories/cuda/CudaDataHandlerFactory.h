#ifndef CUDADATAHANDLERFACTORY_H_
#define CUDADATAHANDLERFACTORY_H_

#include <DataHandlerFactory.h>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <CudaMissingDataHandler.h>
#include <EnvironmentFactorHandler.h>
#include <EnvironmentVector.h>
#include <CudaEnvironmentVector.h>
#include <InteractionVector.h>
#include <PhenotypeVector.h>
#include <PhenotypeHandler.h>
#include <CovariatesMatrix.h>
#include <CovariatesHandler.h>
#include <ContingencyTableFactory.h>
#include <RiskAlleleStrategy.h>
#include <AlleleStatisticsFactory.h>
#include <CudaAlleleStatisticsFactory.h>
#include <CudaContingencyTableFactory.h>
#include <BedReader.h>
#include <Stream.h>

namespace CuEira {
namespace CUDA {

using namespace CuEira::Container;

/*
 * This class....
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaDataHandlerFactory: public DataHandlerFactory<DeviceMatrix, DeviceVector> {
public:
  explicit CudaDataHandlerFactory(const Configuration& configuration, const RiskAlleleStrategy& riskAlleleStrategy,
      Task::DataQueue& dataQueue);
  virtual ~CudaDataHandlerFactory();

  virtual DataHandler<DeviceMatrix, DeviceVector>* constructDataHandler(const Stream& stream,
      const FileIO::BedReader<DeviceVector>* bedReader,
      const EnvironmentFactorHandler<DeviceVector>& environmentFactorHandler,
      const PhenotypeHandler<DeviceVector>& phenotypeHandler,
      const CovariatesHandler<DeviceMatrix>& covariatesHandler) const;

  CudaDataHandlerFactory(const CudaDataHandlerFactory&) = delete;
  CudaDataHandlerFactory(CudaDataHandlerFactory&&) = delete;
  CudaDataHandlerFactory& operator=(const CudaDataHandlerFactory&) = delete;
  CudaDataHandlerFactory& operator=(CudaDataHandlerFactory&&) = delete;

private:

};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* CUDADATAHANDLERFACTORY_H_ */
