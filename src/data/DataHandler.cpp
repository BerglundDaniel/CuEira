#include "DataHandler.h"

namespace CuEira {

DataHandler::DataHandler(StatisticModel statisticModel, const FileIO::BedReader& bedReader,
    const std::vector<const EnvironmentFactor*>& environmentInformation, Task::DataQueue& dataQueue,
    Container::EnvironmentVector* environmentVector) :
    currentRecode(ALL_RISK), dataQueue(dataQueue), statisticModel(statisticModel), bedReader(bedReader), interactionVector(
        nullptr), snpVector(nullptr), environmentVector(environmentVector), environmentInformation(
        environmentInformation), currentSNP(nullptr), currentEnvironmentFactorPos(0), state(NOT_INITIALISED) {

}

DataHandler::~DataHandler() {
  delete interactionVector;
  delete snpVector;
  delete environmentVector;
}

const SNP& DataHandler::getCurrentSNP() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getter run next() at least once.");
  }
#endif
  return snpVector->getAssociatedSNP();
}

const EnvironmentFactor& DataHandler::getCurrentEnvironmentFactor() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw new InvalidState("Before using the getter run next() at least once.");
  }
#endif
  return environmentVector->getCurrentEnvironmentFactor();
}

bool DataHandler::next() {
  if(state == NOT_INITIALISED){
    if(!dataQueue.hasNext()){
      return false;
    }

    state = INITIALISED;
    SNP* nextSNP = dataQueue.next();
    const EnvironmentFactor* nextEnvironmentFactor = environmentInformation[0];
    currentEnvironmentFactorPos = 0;

    environmentVector->switchEnvironmentFactor(*nextEnvironmentFactor);
    readSNP(*nextSNP);

  }else{
    currentRecode = ALL_RISK;
    delete interactionVector;

    if(currentEnvironmentFactorPos == environmentInformation.size() - 1){
      if(!dataQueue.hasNext()){
        return false;
      }

      SNP* nextSNP = dataQueue.next();
      currentEnvironmentFactorPos = 0;

      readSNP(*nextSNP);
    }else{
      currentEnvironmentFactorPos++;
    }

    const EnvironmentFactor* nextEnvironmentFactor = environmentInformation[currentEnvironmentFactorPos];
    environmentVector->switchEnvironmentFactor(*nextEnvironmentFactor);
  } /* else if NOT_INITIALISED */

  interactionVector = new Container::InteractionVector(*environmentVector, *snpVector);

  snpVector->applyStatisticModel(statisticModel, interactionVector->getRecodedData());
  environmentVector->applyStatisticModel(statisticModel, interactionVector->getRecodedData());

  return true;
}

void DataHandler::readSNP(SNP& nextSnp) {
  delete snpVector;

  snpVector = bedReader.readSNP(nextSnp);
  if(!nextSnp.getInclude()){ //SNP can changed based on the reading so we have to check that it still should be included
    this->next(); //FIXME write result or something?
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

  snpVector->recode(recode);

  environmentVector->recode(recode);

  interactionVector->recode();

  snpVector->applyStatisticModel(statisticModel, interactionVector->getRecodedData());
  environmentVector->applyStatisticModel(statisticModel, interactionVector->getRecodedData());
}

const Container::HostVector& DataHandler::getSNP() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getter run next() at least once.");
  }
#endif

  return snpVector->getRecodedData();
}

const Container::HostVector& DataHandler::getInteraction() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getter run next() at least once.");
  }
#endif

  return interactionVector->getRecodedData();
}

const Container::HostVector& DataHandler::getEnvironment() const {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Before using the getter run next() at least once.");
  }
#endif

  return environmentVector->getRecodedData();
}

} /* namespace CuEira */
