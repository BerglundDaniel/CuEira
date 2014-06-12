#include "GpuModelHandler.h"

namespace CuEira {
namespace Model {

GpuModelHandler::GpuModelHandler(DataHandler& dataHandler,
    LogisticRegression::LogisticRegressionConfiguration* logisticRegressionConfiguration) :
    ModelHandler(dataHandler), logisticRegressionConfiguration(logisticRegressionConfiguration), numberOfRows(
        logisticRegressionConfiguration->getNumberOfRows()), numberOfPredictors(
        logisticRegressionConfiguration->getNumberOfPredictors()) {

}

GpuModelHandler::~GpuModelHandler() {
  delete logisticRegressionConfiguration;
}

Statistics* GpuModelHandler::calculateModel() {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Must run next() on ModelHandler before calculateModel().");
  }
#endif

  if(state == INITIALISED_READY){
    logisticRegressionConfiguration->setSNP(*snpData);
    logisticRegressionConfiguration->setEnvironmentFactor(*environmentData);
  }else{

    if(*currentSNP != *oldSNP){
      logisticRegressionConfiguration->setSNP(*snpData);
    }

    if(*currentEnvironmentFactor != *oldEnvironmentFactor){
      logisticRegressionConfiguration->setEnvironmentFactor(*environmentData);
    }
  }

  logisticRegressionConfiguration->setInteraction(*interactionData);

  LogisticRegression::LogisticRegression logisticRegression(logisticRegressionConfiguration);

  const Container::HostVector* betaCoefficents = &logisticRegression.getBeta(); //FIXME
  const Container::HostMatrix& covarianceMatrix = logisticRegression.getCovarianceMatrix();

  //Does any of the data need to be recoded?
  Recode recode;
  double snpBeta = betaCoefficents(1);
  double envBeta = betaCoefficents(2);
  double interactionBeta = betaCoefficents(3);

  if(snpBeta < 0 && snpBeta < envBeta && snpBeta < interactionBeta){
    recode = SNP_PROTECT;
  }else if(envBeta < 0 && envBeta < snpBeta && envBeta < interactionBeta){
    recode = ENVIRONMENT_PROTECT;
  }else if(interactionBeta < 0 && interactionBeta < snpBeta && interactionBeta < envBeta){
    recode = INTERACTION_PROTECT;
  }

  if(recode != ALL_RISK){
    dataHandler.recode(recode);

    //Calculate again
    logisticRegression(logisticRegressionConfiguration);
    betaCoefficents = &logisticRegression.getBeta(); //FIXME
    covarianceMatrix = logisticRegression.getCovarianceMatrix();
  }

  const Container::HostVector* standardError = new PinnedHostVector(numberOfPredictors);
  for(int i = 0; i < numberOfPredictors; ++i){
    (*standardError)(i) = covarianceMatrix(i, i);
  }

  return new Statistics(betaCoefficents, standardError); //FIXME statistics owns beta and stanarderror?
}

} /* namespace Model */
} /* namespace CuEira */
