#include "DataHandler.h"

namespace CuEira {

DataHandler::DataHandler(const Configuration& configuration, const FileIO::BedReader& bedReader,
    const ContingencyTableFactory& contingencyTableFactory,
    const std::vector<const EnvironmentFactor*>& environmentInformation, Task::DataQueue& dataQueue,
    Container::EnvironmentVector* environmentVector, Container::InteractionVector* interactionVector) :
    configuration(configuration), currentRecode(ALL_RISK), dataQueue(&dataQueue), statisticModel(
        configuration.getStatisticModel()), bedReader(&bedReader), interactionVector(interactionVector), snpVector(
        nullptr), environmentVector(environmentVector), environmentInformation(&environmentInformation), currentEnvironmentFactorPos(
        environmentInformation.size() - 1), state(NOT_INITIALISED), contingencyTable(nullptr), contingencyTableFactory(
        &contingencyTableFactory), currentSNP(nullptr), cellCountThreshold(configuration.getCellCountThreshold()), alleleStatistics(
        nullptr) {

}

DataHandler::DataHandler(const Configuration& configuration) :
    configuration(configuration), currentRecode(ALL_RISK), dataQueue(nullptr), statisticModel(ADDITIVE), bedReader(
        nullptr), interactionVector(nullptr), snpVector(nullptr), environmentVector(nullptr), environmentInformation(
        nullptr), currentEnvironmentFactorPos(0), state(NOT_INITIALISED), contingencyTable(nullptr), contingencyTableFactory(
        nullptr), currentSNP(nullptr), alleleStatistics(nullptr), cellCountThreshold(0) {

}

DataHandler::~DataHandler() {
  delete interactionVector;
  delete snpVector;
  delete environmentVector;
  delete contingencyTable;
  delete alleleStatistics;
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

const ContingencyTable& DataHandler::getContingencyTable() const {
  return *contingencyTable;
}

const AlleleStatistics& DataHandler::getAlleleStatistics() const {
  return *alleleStatistics;
}

DataHandlerState DataHandler::next() {
  currentRecode = ALL_RISK;
  if(currentEnvironmentFactorPos == environmentInformation->size() - 1){ //Check if we were at the last EnvironmentFactor so should start with next snp
    delete currentSNP;
    currentSNP = dataQueue->next();
    if(currentSNP == nullptr){
      return DONE;
    }

    bool include = readSNP(*currentSNP);
    if(!include){
      return SKIP;
    }
    currentEnvironmentFactorPos = 0;
  }else{
#ifdef DEBUG
    if(state == NOT_INITIALISED){
      throw InvalidState("This shouldn't happen in DataHandler.");
    }
#endif

    currentEnvironmentFactorPos++;
    snpVector->recode(ALL_RISK);
  }

#ifdef DEBUG
  if(state == NOT_INITIALISED){
    state = INITIALISED;
  }
#endif

  const EnvironmentFactor* nextEnvironmentFactor = (*environmentInformation)[currentEnvironmentFactorPos];
  environmentVector->switchEnvironmentFactor(*nextEnvironmentFactor);
  interactionVector->recode(*snpVector);

  delete contingencyTable;
  contingencyTable = contingencyTableFactory->constructContingencyTable(*snpVector, *environmentVector);

  setSNPInclude(*currentSNP, *contingencyTable);
  if(!currentSNP->shouldInclude()){
    return SKIP;
  }

  snpVector->applyStatisticModel(statisticModel, interactionVector->getRecodedData());
  environmentVector->applyStatisticModel(statisticModel, interactionVector->getRecodedData());

  return CALCULATE;
}

bool DataHandler::readSNP(SNP& nextSnp) {
  delete snpVector;
  delete alleleStatistics;

  std::pair<const AlleleStatistics*, Container::SNPVector*>* pair = bedReader->readSNP(nextSnp);
  alleleStatistics = pair->first;
  snpVector = pair->second;

  delete pair;
  return nextSnp.shouldInclude();
}

Recode DataHandler::getRecode() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getter run next() at least once.");
  }
#endif

  return currentRecode;
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

  snpVector->applyStatisticModel(statisticModel, interactionVector->getRecodedData());
  environmentVector->applyStatisticModel(statisticModel, interactionVector->getRecodedData());
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
