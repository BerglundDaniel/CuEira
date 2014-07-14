#include "GpuModelHandler.h"

namespace CuEira {
namespace Model {

GpuModelHandler::GpuModelHandler(const StatisticsFactory& statisticsFactory, DataHandler* dataHandler,
    LogisticRegression::LogisticRegressionConfiguration& logisticRegressionConfiguration,
    LogisticRegression::LogisticRegression* logisticRegression) :
    ModelHandler(statisticsFactory, dataHandler), logisticRegressionConfiguration(logisticRegressionConfiguration), logisticRegression(
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
  std::cerr << "g1" << std::endl;
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
  std::cerr << "g2" << std::endl;
  logisticRegressionConfiguration.setInteraction(*interactionData);
  LogisticRegression::LogisticRegressionResult* logisticRegressionResult = logisticRegression->calculate();
  std::cerr << "g3" << std::endl;
  Recode recode = logisticRegressionResult->calculateRecode();
  std::cerr << "g4" << std::endl;
  if(recode != ALL_RISK){
    dataHandler->recode(recode);

    logisticRegressionConfiguration.setSNP(*snpData);
    logisticRegressionConfiguration.setEnvironmentFactor(*environmentData);
    logisticRegressionConfiguration.setInteraction(*interactionData);

    //Calculate again
    delete logisticRegressionResult;
    logisticRegressionResult = logisticRegression->calculate();
  }
  std::cerr << "g5" << std::endl;
  return statisticsFactory.constructStatistics(logisticRegressionResult);
}

} /* namespace Model */
} /* namespace CuEira */
