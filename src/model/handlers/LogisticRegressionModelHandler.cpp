#include "LogisticRegressionModelHandler.h"

namespace CuEira {
namespace Model {
namespace LogisticRegression {

LogisticRegressionModelHandler::LogisticRegressionModelHandler(const CombinedResultsFactory& combinedResultsFactory,
    DataHandler* dataHandler,
    CuEira::Model::LogisticRegression::LogisticRegressionConfiguration& logisticRegressionConfiguration,
    LogisticRegression* logisticRegression) :
    ModelHandler(combinedResultsFactory, dataHandler), logisticRegressionConfiguration(logisticRegressionConfiguration), logisticRegression(
        logisticRegression), numberOfRows(logisticRegressionConfiguration.getNumberOfRows()), numberOfPredictors(
        logisticRegressionConfiguration.getNumberOfPredictors()) {

}

LogisticRegressionModelHandler::~LogisticRegressionModelHandler() {
  delete logisticRegression;
}

CombinedResults* LogisticRegressionModelHandler::calculateModel() {
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Must run next() on ModelHandler before calculateModel().");
  }
#endif
  dataHandler->applyStatisticModel(ADDITIVE);

  logisticRegressionConfiguration.setSNP(*snpData);
  logisticRegressionConfiguration.setEnvironmentFactor(*environmentData);
  logisticRegressionConfiguration.setInteraction(*interactionData);

  LogisticRegressionResult* additiveLogisticRegressionResult = logisticRegression->calculate();
  CUDA::handleCudaStatus(cudaGetLastError(), "Error with GpuModelHandler: ");

  dataHandler->applyStatisticModel(MULTIPLICATIVE);

  logisticRegressionConfiguration.setSNP(*snpData);
  logisticRegressionConfiguration.setEnvironmentFactor(*environmentData);
  //Don't need to set the interaction again since it doesn't change between additive and multiplicative model

  LogisticRegressionResult* multiplicativeLogisticRegressionResult = logisticRegression->calculate();
  CUDA::handleCudaStatus(cudaGetLastError(), "Error with GpuModelHandler: ");

  Recode recode = additiveLogisticRegressionResult->calculateRecode();
  if(recode != ALL_RISK){
    dataHandler->recode(recode);
    dataHandler->applyStatisticModel(ADDITIVE);

    logisticRegressionConfiguration.setSNP(*snpData);
    logisticRegressionConfiguration.setEnvironmentFactor(*environmentData);
    logisticRegressionConfiguration.setInteraction(*interactionData);

    //Calculate again
    delete additiveLogisticRegressionResult;
    additiveLogisticRegressionResult = logisticRegression->calculate();
    CUDA::handleCudaStatus(cudaGetLastError(), "Error with GpuModelHandler: ");
  }

  /* FIXME TMP thing due to comparasion with GEISA
   dataHandler->applyStatisticModel(MULTIPLICATIVE);

   logisticRegressionConfiguration.setSNP(*snpData);
   logisticRegressionConfiguration.setEnvironmentFactor(*environmentData);
   //Don't need to set the interaction again since it doesn't change between additive and multiplicative model

   LogisticRegressionResult* multiplicativeLogisticRegressionResult = logisticRegression->calculate();
   CUDA::handleCudaStatus(cudaGetLastError(), "Error with GpuModelHandler: ");
   */
  return combinedResultsFactory.constructCombinedResults(additiveLogisticRegressionResult,
      multiplicativeLogisticRegressionResult, recode);
}

} /* namespace CudaLogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
