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
  ModelStatisticsFactory interactionStatisticsFactory;
  MKLWrapper blasWrapper;
  Model::CombinedResultsFactory combinedResultsFactory(interactionStatisticsFactory);
  const Container::DeviceVector& deviceOutcomes = device->getOutcomes();

  Stream* stream = streamFactory.constructStream(*device);
  KernelWrapper* kernelWrapper = new KernelWrapper(*stream);
  HostToDevice hostToDevice(*stream);
  DeviceToHost deviceToHost(*stream);

  Model::LogisticRegression::CUDA::CudaLogisticRegressionConfiguration* logisticRegressionConfiguration = nullptr;

  if(configuration->covariateFileSpecified()){
    //logisticRegressionConfiguration = new Model::LogisticRegression::CUDA::CudaLogisticRegressionConfiguration(*configuration,
    //hostToDevice, deviceToHost, deviceOutcomes, *kernelWrapper, blasWrapper, *covariates);
  }else{
    logisticRegressionConfiguration = new Model::LogisticRegression::CUDA::CudaLogisticRegressionConfiguration(
        *configuration, hostToDevice, deviceToHost, deviceOutcomes, *kernelWrapper, blasWrapper);
  }

  Model::LogisticRegression::CUDA::CudaLogisticRegression* logisticRegression =
      new Model::LogisticRegression::CUDA::CudaLogisticRegression(logisticRegressionConfiguration);
  Model::ModelHandler* modelHandler = new Model::LogisticRegression::LogisticRegressionModelHandler(
      combinedResultsFactory, dataHandler, *logisticRegressionConfiguration, logisticRegression);
  CUDA::handleCudaStatus(cudaGetLastError(), "Error with initialisation in GPUWorkerThread ");

#ifdef PROFILE
  boost::chrono::system_clock::time_point startCalc = boost::chrono::system_clock::now();
#endif

  DataHandlerState dataHandlerState = modelHandler->next();
  while(dataHandlerState != DONE){
    const Model::ModelInformation& modelInformation = modelHandler->getCurrentModelInformation();

    if(dataHandlerState == SKIP){
      resultWriter->writePartialResult(modelInformation);
    }else{
      Model::CombinedResults* combinedResults = modelHandler->calculateModel();

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
  boost::chrono::duration<double> calcTime = stopPoint - startCalc;

  //std::cerr << "Thread: " << std::this_thread::get_id() << " TotalTime: " << diffThreadSec << std::endl;
  //std::cerr << "Thread: " << std::this_thread::get_id() << " CalcTime: " << calcTime << std::endl;
#endif
}

} /* namespace CUDA */
} /* namespace CuEira */
