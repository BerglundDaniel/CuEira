#include "DataHandler.h"

namespace CuEira {

DataHandler::DataHandler(StatisticModel statisticModel, const FileIO::BedReader& bedReader,
    const std::vector<const EnvironmentFactor*>& environmentInformation, Task::DataQueue& dataQueue,
    Container::EnvironmentVector* environmentVector, Container::InteractionVector* interactionVector) :
    currentRecode(ALL_RISK), dataQueue(&dataQueue), statisticModel(statisticModel), bedReader(&bedReader), interactionVector(
        interactionVector), snpVector(nullptr), environmentVector(environmentVector), environmentInformation(
        &environmentInformation), currentEnvironmentFactorPos(environmentInformation.size() - 1), state(NOT_INITIALISED) {

}

DataHandler::DataHandler() :
    currentRecode(ALL_RISK), dataQueue(nullptr), statisticModel(ADDITIVE), bedReader(nullptr), interactionVector(
        nullptr), snpVector(nullptr), environmentVector(nullptr), environmentInformation(nullptr), currentEnvironmentFactorPos(
        0), state(NOT_INITIALISED) {

}

DataHandler::~DataHandler() {
  delete interactionVector;
  delete snpVector;
  delete environmentVector;
}

const SNP& DataHandler::getCurrentSNP() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getCurrentSNP use next() at least once.");
  }
#endif
  return snpVector->getAssociatedSNP();
}

const EnvironmentFactor& DataHandler::getCurrentEnvironmentFactor() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getCurrentEnvironmentFactor use next() at least once.");
  }
#endif
  return *(*environmentInformation)[currentEnvironmentFactorPos];
}

DataHandlerState DataHandler::next() {
  if(currentEnvironmentFactorPos == environmentInformation->size() - 1){ //Check if we were at the last EnvironmentFactor so should start with next snp
    SNP* nextSNP = dataQueue->next();
    if(nextSNP == nullptr){
      return DONE;
    }

    bool include = readSNP(*nextSNP);
    if(!include){
      return EXCLUDE;
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

  std::cerr << "DataHandler " << snpVector->getAssociatedSNP().getId().getString() << " "
      << (*environmentInformation)[currentEnvironmentFactorPos]->getId().getString() << std::endl;
  const Container::HostVector& snpData = snpVector->getRecodedData();
  const Container::HostVector& environmentData = environmentVector->getRecodedData();

  for(int i = 0; i < snpData.getNumberOfRows(); ++i){
    std::cerr << snpData(i);
  }
  std::cerr << std::endl;

  for(int i = 0; i < snpData.getNumberOfRows(); ++i){
    std::cerr << environmentData(i);
  }
  std::cerr << std::endl;

  interactionVector->recode(*snpVector);

  //TODO check if it should be included based on the interaction stuff, need to store the freqs some where

  const Container::HostVector& interactionData = interactionVector->getRecodedData();
  for(int i = 0; i < snpData.getNumberOfRows(); ++i){
    std::cerr << interactionData(i);
  }
  std::cerr << std::endl;

  snpVector->applyStatisticModel(statisticModel, interactionVector->getRecodedData());
  environmentVector->applyStatisticModel(statisticModel, interactionVector->getRecodedData());

  currentRecode = ALL_RISK;
  return INCLUDE;
}

bool DataHandler::readSNP(SNP& nextSnp) {
  delete snpVector;
  snpVector = bedReader->readSNP(nextSnp);

  return nextSnp.getInclude();
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

} /* namespace CuEira */
