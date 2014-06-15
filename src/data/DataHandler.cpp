#include "DataHandler.h"

namespace CuEira {

DataHandler::DataHandler(StatisticModel statisticModel, const FileIO::BedReader& bedReader,
    const EnvironmentFactorHandler& environmentFactorHandler, Task::DataQueue& dataQueue) :
    currentRecode(ALL_RISK), numberOfIndividualsToInclude(), dataQueue(dataQueue), statisticModel(statisticModel), bedReader(
        bedReader), interactionVector(nullptr), snpVector(nullptr), environmentVector(nullptr), environmentFactorHandler(
        environmentFactorHandler), firstNext(true) {

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
  return snpVector->getAssociatedSNP();
}

const EnvironmentFactor& DataHandler::getCurrentEnvironmentFactor() const {
  return environmentVector->getCurrentEnvironmentFactor();
}

bool DataHandler::next() {
  if(!dataQueue.hasNext()){
    return false;
  }

  std::pair<SNP*, EnvironmentFactor*>* dataPair = dataQueue.next();
  SNP* nextSnp = dataPair->first;
  EnvironmentFactor* nextEnvironmentFactor = dataPair->second;

  delete dataPair;

  if(firstNext){
    firstNext = false;
    readSNP(*nextSnp);
    environmentVector = new Container::EnvironmentVector(environmentFactorHandler, *nextEnvironmentFactor);
    return true;
  }else{
    currentRecode = ALL_RISK;
    delete interactionVector;

    if(!(getCurrentSNP() == *nextSnp)){
      readSNP(*nextSnp);
    }else{
      snpVector->recode(currentRecode);
    }

    if(!(getCurrentEnvironmentFactor() == *nextEnvironmentFactor)){
      environmentVector->switchEnvironmentFactor(*nextEnvironmentFactor);
    }else{
      environmentVector->recode(currentRecode);
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
  return currentRecode;
}

void DataHandler::recode(Recode recode) {
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
  return snpVector->getRecodedData();
}

const Container::HostVector& DataHandler::getInteraction() const {
  return interactionVector->getRecodedData();
}

const Container::HostVector& DataHandler::getEnvironment() const {
  return environmentVector->getRecodedData();
}

} /* namespace CuEira */
