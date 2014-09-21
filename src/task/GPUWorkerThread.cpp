#include "GPUWorkerThread.h"

namespace CuEira {
namespace CUDA {

void GPUWorkerThread(const Configuration* configuration, const Device* device,
    const DataHandlerFactory* dataHandlerFactory, FileIO::ResultWriter* resultWriter) {

  DataHandler* dataHandler = dataHandlerFactory->constructDataHandler();
  device->setActiveDevice();
  CUDA::handleCudaStatus(cudaGetLastError(), "Error before initialisation in GPUWorkerThread ");

  StreamFactory streamFactory;
  InteractionStatisticsFactory interactionStatisticsFactory;
  Model::CombinedResultsFactory combinedResultsFactory(interactionStatisticsFactory);
  const Container::DeviceVector& deviceOutcomes = device->getOutcomes();

  Stream* stream = streamFactory.constructStream(*device);
  KernelWrapper* kernelWrapper = new KernelWrapper(*stream);
  HostToDevice hostToDevice(*stream);
  DeviceToHost deviceToHost(*stream);

  Model::LogisticRegression::LogisticRegressionConfiguration* logisticRegressionConfiguration = nullptr;

  if(configuration->covariateFileSpecified()){
    //logisticRegressionConfiguration = new Model::LogisticRegression::LogisticRegressionConfiguration(*configuration,
    //hostToDevice, deviceOutcomes, *kernelWrapper, *covariates);
  }else{
    logisticRegressionConfiguration = new Model::LogisticRegression::LogisticRegressionConfiguration(*configuration,
        hostToDevice, deviceOutcomes, *kernelWrapper);
  }

  Model::LogisticRegression::LogisticRegression* logisticRegression = new Model::LogisticRegression::LogisticRegression(
      logisticRegressionConfiguration, hostToDevice, deviceToHost);
  Model::ModelHandler* modelHandler = new Model::GpuModelHandler(combinedResultsFactory, dataHandler,
      *logisticRegressionConfiguration, logisticRegression);
  CUDA::handleCudaStatus(cudaGetLastError(), "Error with initialisation in GPUWorkerThread ");

  Model::ModelInformation* modelInformation = modelHandler->next();
  while(modelInformation->getModelState() != DONE){

    if(modelInformation->getModelState() == SKIP){
      resultWriter->writePartialResult(modelInformation);
    }else{
      Model::CombinedResults* combinedResults = modelHandler->calculateModel();

      CUDA::handleCudaStatus(cudaGetLastError(), "Error with ModelHandler in GPUWorkerThread ");

      resultWriter->writeFullResult(modelInformation, combinedResults);
    } //else

    modelInformation = modelHandler->next();
  }

  delete stream;
  delete kernelWrapper;
  delete modelHandler;

}

} /* namespace CUDA */
} /* namespace CuEira */
