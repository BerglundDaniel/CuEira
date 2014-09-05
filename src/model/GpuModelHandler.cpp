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

  logisticRegressionConfiguration.setSNP(*snpData);
  logisticRegressionConfiguration.setEnvironmentFactor(*environmentData);
  logisticRegressionConfiguration.setInteraction(*interactionData);

  std::cerr << "Model" << std::endl;
  std::cerr << currentSNP->getId().getString() << " " << currentEnvironmentFactor->getId().getString() << std::endl;
  for(int i = 0; i < snpData->getNumberOfRows(); ++i){
    std::cerr << (*snpData)(i);
  }
  std::cerr << std::endl;

  for(int i = 0; i < snpData->getNumberOfRows(); ++i){
    std::cerr << (*environmentData)(i);
  }
  std::cerr << std::endl;

  for(int i = 0; i < snpData->getNumberOfRows(); ++i){
    std::cerr << (*interactionData)(i);
  }
  std::cerr << std::endl;

  LogisticRegression::LogisticRegressionResult* logisticRegressionResult = logisticRegression->calculate();
  CUDA::handleCudaStatus(cudaGetLastError(), "Error with GpuModelHandler: ");

  std::cerr << "LR numiter " << logisticRegressionResult->getNumberOfIterations() << std::endl;
  const Container::HostVector& b = logisticRegressionResult->getBeta();
  std::cerr << "LR beta " << b(0) << " " << b(1) << " " << b(2) << " " << b(3) << std::endl;

  Recode recode = logisticRegressionResult->calculateRecode();
  if(recode != ALL_RISK){
    dataHandler->recode(recode);

    logisticRegressionConfiguration.setSNP(*snpData);
    logisticRegressionConfiguration.setEnvironmentFactor(*environmentData);
    logisticRegressionConfiguration.setInteraction(*interactionData);

    //Calculate again
    delete logisticRegressionResult;
    logisticRegressionResult = logisticRegression->calculate();
    CUDA::handleCudaStatus(cudaGetLastError(), "Error with GpuModelHandler: ");
  }

  return statisticsFactory.constructStatistics(logisticRegressionResult);
}

} /* namespace Model */
} /* namespace CuEira */