#include "LogisticRegressionModelHandler.h"

namespace CuEira {
namespace Model {
namespace LogisticRegression {

template<typename Matrix, typename Vector>
LogisticRegressionModelHandler<Matrix, Vector>::LogisticRegressionModelHandler(
    const CombinedResultsFactory& combinedResultsFactory, DataHandler<Matrix, Vector>* dataHandler,
    LogisticRegressionConfiguration& logisticRegressionConfiguration, LogisticRegression* logisticRegression,
    AdditiveInteractionModel<Vector>* additiveInteractionModel,
    MultiplicativeInteractionModel<Vector>* multiplicativeInteractionModel) :
    ModelHandler<Matrix, Vector>(combinedResultsFactory, dataHandler), logisticRegressionConfiguration(
        logisticRegressionConfiguration), logisticRegression(logisticRegression), additiveInteractionModel(
        additiveInteractionModel), multiplicativeInteractionModel(multiplicativeInteractionModel), numberOfRows(
        logisticRegressionConfiguration.getNumberOfRows()), numberOfPredictors(
        logisticRegressionConfiguration.getNumberOfPredictors()){

}

template<typename Matrix, typename Vector>
LogisticRegressionModelHandler<Matrix, Vector>::~LogisticRegressionModelHandler(){
  delete logisticRegression;
  delete additiveInteractionModel;
  delete multiplicativeInteractionModel;
}

template<typename Matrix, typename Vector>
CombinedResults* LogisticRegressionModelHandler<Matrix, Vector>::calculateModel(){
#ifdef DEBUG
  if(state == NOT_INITIALISED){
    throw InvalidState("Must run next() on ModelHandler before calculateModel().");
  }
#endif
  ModelHandler<Matrix, Vector>::dataHandler->applyStatisticModel(additiveInteractionModel);

  logisticRegressionConfiguration.setSNP(*ModelHandler<Matrix, Vector>::snpData);
  logisticRegressionConfiguration.setEnvironmentFactor(*ModelHandler<Matrix, Vector>::environmentData);
  logisticRegressionConfiguration.setInteraction(*ModelHandler<Matrix, Vector>::interactionData);

  LogisticRegressionResult* additiveLogisticRegressionResult = logisticRegression->calculate();
#ifndef CPU
  CUDA::handleCudaStatus(cudaGetLastError(), "Error with GpuModelHandler: ");
#endif
  Recode recode = additiveLogisticRegressionResult->calculateRecode();

  if(recode != ALL_RISK){
    ModelHandler<Matrix, Vector>::dataHandler->recode(recode);
    ModelHandler<Matrix, Vector>::dataHandler->applyStatisticModel(additiveInteractionModel);

    logisticRegressionConfiguration.setSNP(*ModelHandler<Matrix, Vector>::snpData);
    logisticRegressionConfiguration.setEnvironmentFactor(*ModelHandler<Matrix, Vector>::environmentData);
    logisticRegressionConfiguration.setInteraction(*ModelHandler<Matrix, Vector>::interactionData);

    //Calculate again
    delete additiveLogisticRegressionResult;
    additiveLogisticRegressionResult = logisticRegression->calculate();
#ifndef CPU
    CUDA::handleCudaStatus(cudaGetLastError(), "Error with GpuModelHandler: ");
#endif
  }

  ModelHandler<Matrix, Vector>::dataHandler->applyStatisticModel(multiplicativeInteractionModel);

  logisticRegressionConfiguration.setSNP(*ModelHandler<Matrix, Vector>::snpData);
  logisticRegressionConfiguration.setEnvironmentFactor(*ModelHandler<Matrix, Vector>::environmentData);
  //Don't need to set the interaction again since it doesn't change between additive and multiplicative model

  LogisticRegressionResult* multiplicativeLogisticRegressionResult = logisticRegression->calculate();
#ifndef CPU
  CUDA::handleCudaStatus(cudaGetLastError(), "Error with GpuModelHandler: ");
#endif

  return ModelHandler<Matrix, Vector>::combinedResultsFactory.constructCombinedResults(additiveLogisticRegressionResult,
      multiplicativeLogisticRegressionResult, recode);
}

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
