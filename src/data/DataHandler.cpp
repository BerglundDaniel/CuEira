#include "DataHandler.h"

namespace CuEira {

#ifdef PROFILE
//FIXME becomes seperate if using both cpu and gpu because of template, not good
boost::chrono::duration<long long, boost::nano> DataHandler<Matrix, Vector>::timeSpentRecode;
boost::chrono::duration<long long, boost::nano> DataHandler<Matrix, Vector>::timeSpentNext;
boost::chrono::duration<long long, boost::nano> DataHandler<Matrix, Vector>::timeSpentSNPRead;
boost::chrono::duration<long long, boost::nano> DataHandler<Matrix, Vector>::timeSpentStatModel;
#endif

template<typename Matrix, typename Vector>
DataHandler<Matrix, Vector>::DataHandler(const Configuration& configuration, FileIO::BedReader<>& bedReader,
    const ContingencyTableFactory& contingencyTableFactory,
    const Model::ModelInformationFactory* modelInformationFactory,
    const std::vector<const EnvironmentFactor*>& environmentInformation, Task::DataQueue& dataQueue,
    Container::EnvironmentVector<Vector>* environmentVector, Container::InteractionVector<Vector>* interactionVector,
    Container::PhenotypeVector<Vector>* phenotypeVector, Container::CovariatesMatrix<Matrix, Vector>* covariatesMatrix,
    MissingDataHandler<Vector>* missingDataHandler, const AlleleStatisticsFactory<Vector>* alleleStatisticsFactory) :

    configuration(configuration), currentRecode(ALL_RISK), dataQueue(&dataQueue), bedReader(bedReader), interactionVector(
        interactionVector), snpVector(nullptr), environmentVector(environmentVector), state(NOT_INITIALISED), contingencyTable(
        nullptr), contingencyTableFactory(&contingencyTableFactory), modelInformationFactory(modelInformationFactory), currentSNP(
        nullptr), cellCountThreshold(configuration.getCellCountThreshold()), alleleStatistics(nullptr), modelInformation(
        nullptr), environmentFactor(&environmentVector->getEnvironmentFactor()), appliedStatisticModel(false), minorAlleleFrequencyThreshold(
        configuration.getMinorAlleleFrequencyThreshold()), missingDataHandler(missingDataHandler), phenotypeVector(
        phenotypeVector), covariatesMatrix(covariatesMatrix), alleleStatisticsFactory(alleleStatisticsFactory){

}

template<typename Matrix, typename Vector>
DataHandler<Matrix, Vector>::DataHandler(const Configuration& configuration) :
    configuration(configuration), currentRecode(ALL_RISK), dataQueue(nullptr), bedReader(nullptr), interactionVector(
        nullptr), snpVector(nullptr), environmentVector(nullptr), state(NOT_INITIALISED), contingencyTable(nullptr), contingencyTableFactory(
        nullptr), currentSNP(nullptr), alleleStatistics(nullptr), cellCountThreshold(0), modelInformationFactory(
        nullptr), modelInformation(nullptr), environmentFactor(nullptr), appliedStatisticModel(false), minorAlleleFrequencyThreshold(
        0), missingDataHandler(nullptr), phenotypeVector(nullptr), covariatesMatrix(nullptr), alleleStatisticsFactory(
        nullptr){

}

template<typename Matrix, typename Vector>
DataHandler<Matrix, Vector>::~DataHandler(){
  delete covariatesMatrix;
  delete interactionVector;
  delete snpVector;
  delete environmentVector;
  delete phenotypeVector;
  delete contingencyTable;
  delete alleleStatistics;
  delete modelInformation;
  delete currentSNP;
  delete missingDataHandler;
  delete alleleStatisticsFactory;
  delete modelInformationFactory;
}

template<typename Matrix, typename Vector>
DataHandlerState DataHandler<Matrix, Vector>::next(){
#ifdef PROFILE
  boost::chrono::system_clock::time_point before = boost::chrono::system_clock::now();
#endif
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    state = INITIALISED;
  }
#endif

  currentRecode = ALL_RISK;
  appliedStatisticModel = false;

  delete contingencyTable;
  delete modelInformation;
  delete currentSNP;
  delete snpVector;
  delete alleleStatistics;

  contingencyTable = nullptr;
  modelInformation = nullptr;
  currentSNP = nullptr;
  snpVector = nullptr;
  alleleStatistics = nullptr;

  currentSNP = dataQueue->next();
  if(currentSNP == nullptr){
#ifdef PROFILE
    boost::chrono::system_clock::time_point after = boost::chrono::system_clock::now();
    timeSpentNext+=after - before;
#endif

    return DONE;
  }

#ifdef PROFILE
  boost::chrono::system_clock::time_point beforeRead = boost::chrono::system_clock::now();
#endif
  snpVector = bedReader->readSNP(*currentSNP);
#ifdef PROFILE
  boost::chrono::system_clock::time_point afterRead = boost::chrono::system_clock::now();
  timeSpentSNPRead+=afterRead - beforeRead;
#endif

  if(snpVector->hasMissing()){
    missingDataHandler->setMissing(snpVector->getMissing());

    environmentVector->recode(currentRecode, *missingDataHandler);
    phenotypeVector->applyMissing(*missingDataHandler);
    covariatesMatrix->applyMissing(*missingDataHandler);
  }else{
    environmentVector->recode(currentRecode);
    phenotypeVector->applyMissing();
    covariatesMatrix->applyMissing();
  }

  alleleStatistics = alleleStatisticsFactory->constructAlleleStatistics(*snpVector, *phenotypeVector);
  RiskAllele riskAllele = riskAlleleStrategy->calculateRiskAllele(*alleleStatistics);
  currentSNP->setRiskAllele(riskAllele);
  snpVector->recode(currentRecode);

  if(alleleStatistics->getMinorAlleleFrequecy() < minorAlleleFrequencyThreshold){
    currentSNP->setInclude(LOW_MAF);
  }

  if(environmentFactor->getVariableType() == BINARY){
    contingencyTable = contingencyTableFactory->constructContingencyTable(*snpVector, *environmentVector);
    modelInformation = modelInformationFactory->constructModelInformation(*currentSNP, *environmentFactor,
        *alleleStatistics, *contingencyTable);

    const std::vector<int>& table = contingencyTable->getTable();
    const int size = table.size();
    for(int i = 0; i < size; ++i){
      if(table[i] < cellCountThreshold){
        currentSNP->setInclude(LOW_CELL_NUMBER);
        break;
      }
    }

  }else{ //if environmentFactor binary
    modelInformation = modelInformationFactory->constructModelInformation(*currentSNP, *environmentFactor,
        *alleleStatistics);
  }

#ifdef PROFILE
  boost::chrono::system_clock::time_point after = boost::chrono::system_clock::now();
  timeSpentNext+=after - before;
#endif

  if(!currentSNP->shouldInclude()){
    return SKIP;
  }else{
    return CALCULATE;
  }
}

template<typename Matrix, typename Vector>
void DataHandler<Matrix, Vector>::applyStatisticModel(const InteractionModel<Vector>& interactionModel){
#ifdef PROFILE
  boost::chrono::system_clock::time_point before = boost::chrono::system_clock::now();
#endif

#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using applyStatisticModel run next() at least once.");
  }
#endif

  if(appliedStatisticModel){ // If a model has been previously applied after a next or a recode then things can have changed
    snpVector->recode(currentRecode);
    environmentVector->recode(currentRecode);
  }

  interactionModel.applyModel(*snpVector, *environmentVector, *interactionVector);

  appliedStatisticModel = true;

#ifdef PROFILE
  boost::chrono::system_clock::time_point after = boost::chrono::system_clock::now();
  timeSpentStatModel+=after - before;
#endif
}

template<typename Matrix, typename Vector>
void DataHandler<Matrix, Vector>::recode(Recode recode){
#ifdef PROFILE
  boost::chrono::system_clock::time_point before = boost::chrono::system_clock::now();
#endif

#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using recode run next() at least once.");
  }
#endif

  if(recode == currentRecode){
#ifdef PROFILE
    boost::chrono::system_clock::time_point after = boost::chrono::system_clock::now();
    timeSpentRecode+=after - before;
#endif

    return;
  }
#ifdef DEBUG
  else if(!(recode == SNP_PROTECT || recode == ENVIRONMENT_PROTECT || recode == INTERACTION_PROTECT || recode == ALL_RISK)){
    throw InvalidState("Unknown recode for DataHandler.");
  }
#endif
  currentRecode = recode;
  appliedStatisticModel = false;

  snpVector->recode(recode);
  environmentVector->recode(recode);

  if(environmentFactor->getVariableType() == BINARY){
    delete contingencyTable;
    delete modelInformation;
    contingencyTable = contingencyTableFactory->constructContingencyTable(*snpVector, *environmentVector);

    modelInformation = modelInformationFactory->constructModelInformation(*currentSNP, *environmentFactor,
        *alleleStatistics, *contingencyTable);
  }

#ifdef PROFILE
  boost::chrono::system_clock::time_point after = boost::chrono::system_clock::now();
  timeSpentRecode+=after - before;
#endif
}

template<typename Matrix, typename Vector>
Recode DataHandler<Matrix, Vector>::getRecode() const{
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getter run next() at least once.");
  }
#endif

  return currentRecode;
}

template<typename Matrix, typename Vector>
const Model::ModelInformation& DataHandler<Matrix, Vector>::getCurrentModelInformation() const{
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using getModelInformation run next() at least once.");
  }
#endif
  return *modelInformation;
}

template<typename Matrix, typename Vector>
const SNP& DataHandler<Matrix, Vector>::getCurrentSNP() const{
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getCurrentSNP use next() at least once.");
  }
#endif
  return *currentSNP;
}

template<typename Matrix, typename Vector>
const EnvironmentFactor& DataHandler<Matrix, Vector>::getCurrentEnvironmentFactor() const{
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getCurrentEnvironmentFactor use next() at least once.");
  }
#endif
  return *environmentFactor;
}

template<typename Matrix, typename Vector>
const Container::SNPVector<Vector>& DataHandler<Matrix, Vector>::getSNPVector() const{
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using getSNPVector run next() at least once.");
  }
#endif
  return *snpVector;
}

template<typename Matrix, typename Vector>
const Container::InteractionVector<Vector>& DataHandler<Matrix, Vector>::getInteractionVector() const{
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using getInteractionVector run next() at least once.");
  }
#endif
  return *interactionVector;
}

template<typename Matrix, typename Vector>
const Container::EnvironmentVector<Vector>& DataHandler<Matrix, Vector>::getEnvironmentVector() const{
  return *environmentVector;
}

template<typename Matrix, typename Vector>
const Container::CovariatesMatrix<Matrix, Vector>& DataHandler<Matrix, Vector>::getCovariatesMatrix() const{
  return *covariatesMatrix;
}

} /* namespace CuEira */
