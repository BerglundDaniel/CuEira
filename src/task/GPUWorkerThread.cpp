#include "GPUWorkerThread.h"

namespace CuEira {
namespace CUDA {

void GPUWorkerThread(const Configuration* configuration, const Device* device,
    const DataHandlerFactory* dataHandlerFactory, FileIO::ResultWriter* resultWriter) {
#ifdef PROFILE
  boost::chrono::system_clock::time_point startPoint = boost::chrono::system_clock::now();
#endif
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

  DataHandlerState dataHandlerState = modelHandler->next();
  while(dataHandlerState != DONE){

    if(dataHandlerState == SKIP){
      const Model::ModelInformation& modelInformation = modelHandler->getCurrentModelInformation();
      resultWriter->writePartialResult(modelInformation);
    }else{
      Model::CombinedResults* combinedResults = modelHandler->calculateModel();
      const Model::ModelInformation& modelInformation = modelHandler->getCurrentModelInformation();

      CUDA::handleCudaStatus(cudaGetLastError(), "Error with ModelHandler in GPUWorkerThread ");

      resultWriter->writeFullResult(modelInformation, combinedResults);
    } //else

    dataHandlerState = modelHandler->next();
  }

  delete stream;
  delete kernelWrapper;
  delete modelHandler;

#ifdef PROFILE
  boost::chrono::system_clock::time_point stopPoint = boost::chrono::system_clock::now();
  boost::chrono::duration<double> diffThreadSec = stopPoint - startPoint;

  std::cerr << "Time for thread " << std::this_thread::get_id() << ": " << diffThreadSec << std::endl;
#endif
}

} /* namespace CUDA */
} /* namespace CuEira */
