#include "DataHandler.h"

namespace CuEira {
namespace Container {

DataHandler::DataHandler(StatisticModel statisticModel, const FileIO::DataFilesReader& dataFilesReader) :
    currentRecode(ALL_RISK), numberOfIndividualsToInclude(), statisticModel(statisticModel), dataFilesReader(
        dataFilesReader) {

}

DataHandler::~DataHandler() {
  delete interactionVector;
  delete snpVector;
  delete environmentVector;
}

int DataHandler::getNumberOfIndividualsToInclude() const {
  return numberOfIndividualsToInclude;
}

const SNP& DataHandler::getAssociatedSNP() const {
  return snpVector->getAssociatedSNP();
}

bool DataHandler::hasNext() const {
  //TODO
}

void DataHandler::next() {
  currentRecode = ALL_RISK;
  delete interactionVector;

  if(){ //TODO if next snp
    delete snpVector;

    SNP snp; //FIXME
    snpVector = dataFilesReader.readSNP(snp);
  }

  if(){ //TODO if next env
    delete environmentVector;

    EnvironmentFactor environmentFactor; //FIXME
    environmentVector = dataFilesReader.getEnvironmentFactor(environmentFactor);
  }

  interactionVector = new InteracionVector(*environmentVector, *snpVector);
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

} /* namespace Container */
} /* namespace CuEira */
