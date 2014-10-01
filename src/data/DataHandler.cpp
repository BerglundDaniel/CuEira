#include "DataHandler.h"

namespace CuEira {

DataHandler::DataHandler(const Configuration& configuration, FileIO::BedReader& bedReader,
    const ContingencyTableFactory& contingencyTableFactory,
    const Model::ModelInformationFactory& modelInformationFactory,
    const std::vector<const EnvironmentFactor*>& environmentInformation, Task::DataQueue& dataQueue,
    Container::EnvironmentVector* environmentVector, Container::InteractionVector* interactionVector) :
    configuration(configuration), currentRecode(ALL_RISK), dataQueue(&dataQueue), bedReader(&bedReader), interactionVector(
        interactionVector), snpVector(nullptr), environmentVector(environmentVector), environmentInformation(
        &environmentInformation), currentEnvironmentFactorPos(environmentInformation.size() - 1), state(
        NOT_INITIALISED), contingencyTable(nullptr), contingencyTableFactory(&contingencyTableFactory), modelInformationFactory(
        &modelInformationFactory), currentSNP(nullptr), cellCountThreshold(configuration.getCellCountThreshold()), alleleStatistics(
        nullptr), modelInformation(nullptr), currentEnvironmentFactor(nullptr) {

}

DataHandler::DataHandler(const Configuration& configuration) :
    configuration(configuration), currentRecode(ALL_RISK), dataQueue(nullptr), bedReader(nullptr), interactionVector(
        nullptr), snpVector(nullptr), environmentVector(nullptr), environmentInformation(nullptr), currentEnvironmentFactorPos(
        0), state(NOT_INITIALISED), contingencyTable(nullptr), contingencyTableFactory(nullptr), currentSNP(nullptr), alleleStatistics(
        nullptr), cellCountThreshold(0), modelInformationFactory(nullptr), modelInformation(nullptr), currentEnvironmentFactor(
        nullptr) {

}

DataHandler::~DataHandler() {
  delete interactionVector;
  delete snpVector;
  delete environmentVector;
  delete contingencyTable;
  delete alleleStatistics;
  delete modelInformation;
  delete currentSNP;
}

DataHandlerState DataHandler::next() {
  currentRecode = ALL_RISK;

  delete contingencyTable;
  delete modelInformation;

  contingencyTable = nullptr;
  modelInformation = nullptr;
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    state = INITIALISED;
  }
#endif

  if(currentEnvironmentFactorPos == environmentInformation->size() - 1){ //Check if we were at the last EnvironmentFactor so should start with next snp
    delete currentSNP;
    delete snpVector;
    delete alleleStatistics;
    currentSNP = nullptr;
    snpVector = nullptr;
    alleleStatistics = nullptr;

    currentSNP = dataQueue->next();
    if(currentSNP == nullptr){
      return DONE;
    }

    bool include = readSNP(*currentSNP);
    if(!include){
      modelInformation = modelInformationFactory->constructModelInformation(*currentSNP, *(*environmentInformation)[0],
          *alleleStatistics);
      return SKIP;
    }
    currentEnvironmentFactorPos = 0;
  }else{
    currentEnvironmentFactorPos++;
    snpVector->recode(ALL_RISK);
  }

  currentEnvironmentFactor = (*environmentInformation)[currentEnvironmentFactorPos];
  environmentVector->switchEnvironmentFactor(*currentEnvironmentFactor);
  interactionVector->recode(*snpVector);

  if(currentEnvironmentFactor->getVariableType() == BINARY){
    contingencyTable = contingencyTableFactory->constructContingencyTable(*snpVector, *environmentVector);

    setSNPInclude(*currentSNP, *contingencyTable);
    if(!currentSNP->shouldInclude()){
      modelInformation = modelInformationFactory->constructModelInformation(*currentSNP, *currentEnvironmentFactor,
          *alleleStatistics, *contingencyTable);
      return SKIP;
    }
  }

  if(currentEnvironmentFactor->getVariableType() == BINARY){
    modelInformation = modelInformationFactory->constructModelInformation(*currentSNP, *currentEnvironmentFactor,
        *alleleStatistics, *contingencyTable);
  }else{
    modelInformation = modelInformationFactory->constructModelInformation(*currentSNP, *currentEnvironmentFactor,
        *alleleStatistics);
  }

  return CALCULATE;
}

void DataHandler::applyStatisticModel(StatisticModel statisticModel) {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using applyStatisticModel run next() at least once.");
  }
#endif
  snpVector->applyStatisticModel(statisticModel, interactionVector->getRecodedData());
  environmentVector->applyStatisticModel(statisticModel, interactionVector->getRecodedData());
}

bool DataHandler::readSNP(SNP& nextSnp) {
  std::pair<const AlleleStatistics*, Container::SNPVector*>* pair = bedReader->readSNP(nextSnp);
  alleleStatistics = pair->first;
  snpVector = pair->second;

  delete pair;
  return nextSnp.shouldInclude();
}

void DataHandler::recode(Recode recode) {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getter run next() at least once.");
  }
#endif

  if(recode == currentRecode){
    return;
  }
#ifdef DEBUG
  else if(!(recode == SNP_PROTECT || recode == ENVIRONMENT_PROTECT || recode == INTERACTION_PROTECT || recode == ALL_RISK)){
    throw InvalidState("Unknown recode for a SNPVector.");
  }
#endif
  currentRecode = recode;

  snpVector->recode(recode);
  environmentVector->recode(recode);
  interactionVector->recode(*snpVector);

  if(currentEnvironmentFactor->getVariableType() == BINARY){
    delete contingencyTable;
    delete modelInformation;
    contingencyTable = contingencyTableFactory->constructContingencyTable(*snpVector, *environmentVector);

    modelInformation = modelInformationFactory->constructModelInformation(*currentSNP, *currentEnvironmentFactor,
        *alleleStatistics, *contingencyTable);
  }
}

Recode DataHandler::getRecode() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getter run next() at least once.");
  }
#endif

  return currentRecode;
}

const Model::ModelInformation& DataHandler::getCurrentModelInformation() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using getModelInformation run next() at least once.");
  }
#endif
  return *modelInformation;
}

const SNP& DataHandler::getCurrentSNP() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getCurrentSNP use next() at least once.");
  }
#endif
  return *currentSNP;
}

const EnvironmentFactor& DataHandler::getCurrentEnvironmentFactor() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getCurrentEnvironmentFactor use next() at least once.");
  }
#endif
  return *(*environmentInformation)[currentEnvironmentFactorPos];
}

const Container::SNPVector& DataHandler::getSNPVector() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using getSNPVector run next() at least once.");
  }
#endif
  return *snpVector;
}

const Container::InteractionVector& DataHandler::getInteractionVector() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using getInteractionVector run next() at least once.");
  }
#endif
  return *interactionVector;
}

const Container::EnvironmentVector& DataHandler::getEnvironmentVector() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using getEnvironmentVector run next() at least once.");
  }
#endif
  return *environmentVector;
}

void DataHandler::setSNPInclude(SNP& snp, const ContingencyTable& contingencyTable) const {
  const std::vector<int>& table = contingencyTable.getTable();
  const int size = table.size();
  for(int i = 0; i < size; ++i){
    if(table[i] <= cellCountThreshold){
      snp.setInclude(LOW_CELL_NUMBER);
      break;
    }
  }
}

} /* namespace CuEira */
