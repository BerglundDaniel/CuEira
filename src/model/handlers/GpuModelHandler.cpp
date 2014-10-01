#include "GpuModelHandler.h"

namespace CuEira {
namespace Model {

GpuModelHandler::GpuModelHandler(const CombinedResultsFactory& combinedResultsFactory, DataHandler* dataHandler,
    LogisticRegression::LogisticRegressionConfiguration& logisticRegressionConfiguration,
    LogisticRegression::LogisticRegression* logisticRegression) :
    ModelHandler(combinedResultsFactory, dataHandler), logisticRegressionConfiguration(logisticRegressionConfiguration), logisticRegression(
        logisticRegression), numberOfRows(logisticRegressionConfiguration.getNumberOfRows()), numberOfPredictors(
        logisticRegressionConfiguration.getNumberOfPredictors()) {

}

GpuModelHandler::~GpuModelHandler() {
  delete logisticRegression;
}

CombinedResults* GpuModelHandler::calculateModel() {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Must run next() on ModelHandler before calculateModel().");
  }
#endif

  logisticRegressionConfiguration.setSNP(*snpData);
  logisticRegressionConfiguration.setEnvironmentFactor(*environmentData);
  logisticRegressionConfiguration.setInteraction(*interactionData);

  LogisticRegression::LogisticRegressionResult* additiveLogisticRegressionResult = logisticRegression->calculate();
  CUDA::handleCudaStatus(cudaGetLastError(), "Error with GpuModelHandler: ");

  Recode recode = additiveLogisticRegressionResult->calculateRecode();
  if(recode != ALL_RISK){
    dataHandler->recode(recode);

    logisticRegressionConfiguration.setSNP(*snpData);
    logisticRegressionConfiguration.setEnvironmentFactor(*environmentData);
    logisticRegressionConfiguration.setInteraction(*interactionData);

    //Calculate again
    delete additiveLogisticRegressionResult;
    additiveLogisticRegressionResult = logisticRegression->calculate();
    CUDA::handleCudaStatus(cudaGetLastError(), "Error with GpuModelHandler: ");
  }

  //TODO do something with dataHandler

  logisticRegressionConfiguration.setSNP(*snpData);
  logisticRegressionConfiguration.setEnvironmentFactor(*environmentData);

  LogisticRegression::LogisticRegressionResult* multiplicativeLogisticRegressionResult =
      logisticRegression->calculate();

  return combinedResultsFactory.constructCombinedResults(additiveLogisticRegressionResult, recode);
}

} /* namespace Model */
} /* namespace CuEira */
