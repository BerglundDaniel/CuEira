#include "CpuDataHandlerFactory.h"

namespace CuEira {
namespace CPU {

CpuDataHandlerFactory::CpuDataHandlerFactory(const Configuration& configuration,
    const RiskAlleleStrategy& riskAlleleStrategy, Task::DataQueue& dataQueue) :
    DataHandlerFactory(configuration, riskAlleleStrategy, dataQueue){

}

CpuDataHandlerFactory::~CpuDataHandlerFactory(){

}

DataHandler<RegularHostMatrix, RegularHostVector>* CpuDataHandlerFactory::constructDataHandler(
    const FileIO::BedReader<RegularHostVector>* bedReader,
    const EnvironmentFactorHandler<RegularHostVector>& environmentFactorHandler,
    const PhenotypeHandler<RegularHostVector>& phenotypeHandler,
    const CovariatesHandler<RegularHostMatrix>& covariatesHandler) const{
  const int numberOfIndividualsTotal = environmentFactorHandler.getNumberOfIndividualsTotal();

  EnvironmentVector<RegularHostVector>* environmentVector = new Container::CPU::CpuEnvironmentVector(
      environmentFactorHandler);
  InteractionVector<RegularHostVector>* interactionVector = new InteractionVector<RegularHostVector>(
      numberOfIndividualsTotal);
  PhenotypeVector<RegularHostVector>* phenotypeVector = new PhenotypeVector<RegularHostVector>(phenotypeHandler);
  CovariatesMatrix<RegularHostMatrix, RegularHostVector>* covariatesMatrix = new CovariatesMatrix<RegularHostMatrix,
      RegularHostVector>(covariatesHandler);

  CpuMissingDataHandler* cpuMissingDataHandler = new CpuMissingDataHandler(numberOfIndividualsTotal);

  const Model::ModelInformationFactory* modelInformationFactory = new Model::ModelInformationFactory();
  const ContingencyTableFactory<RegularHostVector>* contingencyTableFactory = new CpuContingencyTableFactory();
  const AlleleStatisticsFactory<RegularHostVector>* alleleStatisticsFactory = new CpuAlleleStatisticsFactory();

  return new DataHandler<RegularHostMatrix, RegularHostVector>(configuration, dataQueue, riskAlleleStrategy, bedReader,
      contingencyTableFactory, modelInformationFactory, environmentVector, interactionVector, phenotypeVector,
      covariatesMatrix, cpuMissingDataHandler, alleleStatisticsFactory);
}

} /* namespace CPU */
} /* namespace CuEira */
