#include "GpuModelHandler.h"

namespace CuEira {
namespace Model {

GpuModelHandler::GpuModelHandler(DataHandler* dataHandler,
    LogisticRegression::LogisticRegressionConfiguration& logisticRegressionConfiguration,
    LogisticRegression::LogisticRegression* logisticRegression) :
    ModelHandler(dataHandler), logisticRegressionConfiguration(logisticRegressionConfiguration), logisticRegression(
        logisticRegression), numberOfRows(logisticRegressionConfiguration.getNumberOfRows()), numberOfPredictors(
        logisticRegressionConfiguration.getNumberOfPredictors()) {

}

GpuModelHandler::~GpuModelHandler() {
  delete logisticRegression;
}

Statistics* GpuModelHandler::calculateModel() {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Must run next() on ModelHandler before calculateModel().");
  }
#endif

  if(state == INITIALISED_READY){
    logisticRegressionConfiguration.setSNP(*snpData);
    logisticRegressionConfiguration.setEnvironmentFactor(*environmentData);
  }else{
    if(!(*currentSNP == *oldSNP)){
      logisticRegressionConfiguration.setSNP(*snpData);
    }

    if(!(*currentEnvironmentFactor == *oldEnvironmentFactor)){
      logisticRegressionConfiguration.setEnvironmentFactor(*environmentData);
    }
  }

  logisticRegressionConfiguration.setInteraction(*interactionData);

  LogisticRegression::LogisticRegressionResult* logisticRegressionResult = logisticRegression->calculate();

  Recode recode = logisticRegressionResult->calculateRecode();

  if(recode != ALL_RISK){
    dataHandler->recode(recode);

    logisticRegressionConfiguration.setSNP(*snpData);
    logisticRegressionConfiguration.setEnvironmentFactor(*environmentData);
    logisticRegressionConfiguration.setInteraction(*interactionData);

    //Calculate again
    delete logisticRegressionResult;
    logisticRegressionResult = logisticRegression->calculate();
  }

  return new Statistics(logisticRegressionResult);
}

} /* namespace Model */
} /* namespace CuEira */
