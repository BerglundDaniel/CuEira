#include "LogisticRegressionModelHandler.h"

namespace CuEira {
namespace Model {
namespace LogisticRegression {

template<typename Matrix, typename Vector>
LogisticRegressionModelHandler<Matrix, Vector>::LogisticRegressionModelHandler(
    const CombinedResultsFactory& combinedResultsFactory, DataHandler<Matrix, Vector>* dataHandler,
    CuEira::Model::LogisticRegression::LogisticRegressionConfiguration& logisticRegressionConfiguration,
    LogisticRegression* logisticRegression) :
    ModelHandler(combinedResultsFactory, dataHandler), logisticRegressionConfiguration(logisticRegressionConfiguration), logisticRegression(
        logisticRegression), numberOfRows(logisticRegressionConfiguration.getNumberOfRows()), numberOfPredictors(
        logisticRegressionConfiguration.getNumberOfPredictors()){

}

template<typename Matrix, typename Vector>
LogisticRegressionModelHandler<Matrix, Vector>::~LogisticRegressionModelHandler(){
  delete logisticRegression;
}

template<typename Matrix, typename Vector>
CombinedResults* LogisticRegressionModelHandler<Matrix, Vector>::calculateModel(){
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Must run next() on ModelHandler before calculateModel().");
  }
#endif
  dataHandler->applyStatisticModel(ADDITIVE); //FIXME

  logisticRegressionConfiguration.setSNP(*snpData);
  logisticRegressionConfiguration.setEnvironmentFactor(*environmentData);
  logisticRegressionConfiguration.setInteraction(*interactionData);

  LogisticRegressionResult* additiveLogisticRegressionResult = logisticRegression->calculate();
#ifndef CPU
  CUDA::handleCudaStatus(cudaGetLastError(), "Error with GpuModelHandler: ");
#endif
  Recode recode = additiveLogisticRegressionResult->calculateRecode();

  if(recode != ALL_RISK){
    dataHandler->recode(recode);
    dataHandler->applyStatisticModel(ADDITIVE); //FIXME

    logisticRegressionConfiguration.setSNP(*snpData);
    logisticRegressionConfiguration.setEnvironmentFactor(*environmentData);
    logisticRegressionConfiguration.setInteraction(*interactionData);

    //Calculate again
    delete additiveLogisticRegressionResult;
    additiveLogisticRegressionResult = logisticRegression->calculate();
#ifndef CPU
  CUDA::handleCudaStatus(cudaGetLastError(), "Error with GpuModelHandler: ");
#endif
  }

  dataHandler->applyStatisticModel(MULTIPLICATIVE); //FIXME

  logisticRegressionConfiguration.setSNP(*snpData);
  logisticRegressionConfiguration.setEnvironmentFactor(*environmentData);
  //Don't need to set the interaction again since it doesn't change between additive and multiplicative model

  LogisticRegressionResult* multiplicativeLogisticRegressionResult = logisticRegression->calculate();
#ifndef CPU
  CUDA::handleCudaStatus(cudaGetLastError(), "Error with GpuModelHandler: ");
#endif

  return combinedResultsFactory.constructCombinedResults(additiveLogisticRegressionResult,
      multiplicativeLogisticRegressionResult, recode);
}

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
