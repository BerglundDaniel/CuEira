#include "DataHandler.h"

namespace CuEira {

DataHandler::DataHandler(StatisticModel statisticModel, const FileIO::BedReader& bedReader,
    const EnvironmentFactorHandler& environmentFactorHandler, Task::DataQueue& dataQueue) :
    currentRecode(ALL_RISK), numberOfIndividualsToInclude(), dataQueue(dataQueue), statisticModel(statisticModel), bedReader(
        bedReader), interactionVector(nullptr), snpVector(nullptr), environmentVector(nullptr), environmentFactorHandler(
        environmentFactorHandler), environmentInformation(environmentFactorHandler.getHeaders()), currentSNP(nullptr), currentEnvironmentFactorPos(
        0), state(NOT_INITIALISED) {

}

DataHandler::~DataHandler() {
  delete interactionVector;
  delete snpVector;
  delete environmentVector;
}

int DataHandler::getNumberOfIndividualsToInclude() const {
  return numberOfIndividualsToInclude;
}

const SNP& DataHandler::getCurrentSNP() const {
  if(state == NOT_INITIALISED){
    throw new InvalidState("Before using the getter run next() at least once.");
  }
  return snpVector->getAssociatedSNP();
}

const EnvironmentFactor& DataHandler::getCurrentEnvironmentFactor() const {
  if(state == NOT_INITIALISED){
    throw new InvalidState("Before using the getter run next() at least once.");
  }
  return environmentVector->getCurrentEnvironmentFactor();
}

bool DataHandler::next() {
  if(state == NOT_INITIALISED){
    state = INITIALISED;
    if(!dataQueue.hasNext()){
      return false;
    }
    SNP* nextSNP = dataQueue.next();
    EnvironmentFactor* nextEnvironmentFactor = environmentInformation[0];

    readSNP(*nextSNP);
    environmentVector = new Container::EnvironmentVector(environmentFactorHandler, *nextEnvironmentFactor);

  }else{
    currentRecode = ALL_RISK;
    delete interactionVector;

    SNP* nextSNP;
    EnvironmentFactor* nextEnvironmentFactor;
    if(currentEnvironmentFactorPos == environmentInformation.size() - 1){
      if(!dataQueue.hasNext()){
        return false;
      }

      nextSNP = dataQueue.next();
      nextEnvironmentFactor = environmentInformation[0];
      currentEnvironmentFactorPos = 0;
    }else{
      currentEnvironmentFactorPos++;
      nextSNP = currentSNP;
      nextEnvironmentFactor = environmentInformation[currentEnvironmentFactorPos];
    }

  } /* else firstNext */

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
  if(state == NOT_INITIALISED){
    throw new InvalidState("Before using the getter run next() at least once.");
  }
  return currentRecode;
}

void DataHandler::recode(Recode recode) {
  if(state == NOT_INITIALISED){
    throw new InvalidState("Before using the getter run next() at least once.");
  }
  if(recode == currentRecode){
    return;
  }else if(!(recode == SNP_PROTECT || recode == ENVIRONMENT_PROTECT || recode == INTERACTION_PROTECT)){
    throw InvalidState("Unknown recode for a SNPVector.");
  }

  if(recode == SNP_PROTECT || recode == INTERACTION_PROTECT){
    snpVector->recode(recode);
  }

  if(recode == ENVIRONMENT_PROTECT || recode == INTERACTION_PROTECT){
    environmentVector->recode(recode);
  }

  interactionVector->recode();

  snpVector->applyStatisticModel(statisticModel, interactionVector->getRecodedData());
  environmentVector->applyStatisticModel(statisticModel, interactionVector->getRecodedData());
}

const Container::HostVector& DataHandler::getSNP() const {
  if(state == NOT_INITIALISED){
    throw new InvalidState("Before using the getter run next() at least once.");
  }
  return snpVector->getRecodedData();
}

const Container::HostVector& DataHandler::getInteraction() const {
  if(state == NOT_INITIALISED){
    throw new InvalidState("Before using the getter run next() at least once.");
  }
  return interactionVector->getRecodedData();
}

const Container::HostVector& DataHandler::getEnvironment() const {
  if(state == NOT_INITIALISED){
    throw new InvalidState("Before using the getter run next() at least once.");
  }
  return environmentVector->getRecodedData();
}

} /* namespace CuEira */
