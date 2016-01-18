#include "CudaDataHandlerFactory.h"

namespace CuEira {
namespace CUDA {

CudaDataHandlerFactory::CudaDataHandlerFactory(const Configuration& configuration,
    const RiskAlleleStrategy& riskAlleleStrategy, Task::DataQueue& dataQueue) :
    DataHandlerFactory(configuration, riskAlleleStrategy, dataQueue){

}

CudaDataHandlerFactory::~CudaDataHandlerFactory(){

}

DataHandler<DeviceMatrix, DeviceVector>* CudaDataHandlerFactory::constructDataHandler(const Stream& stream,
    const FileIO::BedReader<DeviceVector>* bedReader,
    const EnvironmentFactorHandler<DeviceVector>& environmentFactorHandler,
    const PhenotypeHandler<DeviceVector>& phenotypeHandler,
    const CovariatesHandler<DeviceMatrix>& covariatesHandler) const{
  const int numberOfIndividualsTotal = environmentFactorHandler.getNumberOfIndividualsTotal();

  EnvironmentVector<DeviceVector>* environmentVector = new Container::CUDA::CudaEnvironmentVector(
      environmentFactorHandler, stream);
  InteractionVector<DeviceVector>* interactionVector = new InteractionVector<DeviceVector>(numberOfIndividualsTotal);
  PhenotypeVector<DeviceVector>* phenotypeVector = new PhenotypeVector<DeviceVector>(phenotypeHandler);
  CovariatesMatrix<DeviceMatrix, DeviceVector>* covariatesMatrix = new CovariatesMatrix<DeviceMatrix, DeviceVector>(
      covariatesHandler);

  CudaMissingDataHandler* missingDataHandler = new CudaMissingDataHandler(numberOfIndividualsTotal, stream);

  const Model::ModelInformationFactory* modelInformationFactory = new Model::ModelInformationFactory();
  const CudaContingencyTableFactory* contingencyTableFactory = new CudaContingencyTableFactory();
  const CudaAlleleStatisticsFactory* alleleStatisticsFactory = new CudaAlleleStatisticsFactory(stream);

  return new DataHandler<DeviceMatrix, DeviceVector>(configuration, dataQueue, riskAlleleStrategy, bedReader,
      contingencyTableFactory, modelInformationFactory, environmentVector, interactionVector, phenotypeVector,
      covariatesMatrix, missingDataHandler, alleleStatisticsFactory);
}

} /* namespace CUDA */
} /* namespace CuEira */
