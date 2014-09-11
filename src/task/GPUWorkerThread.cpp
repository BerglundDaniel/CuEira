#include "GPUWorkerThread.h"

namespace CuEira {
namespace CUDA {

void GPUWorkerThread(const Configuration* configuration, const Device* device,
    const DataHandlerFactory* dataHandlerFactory, const FileIO::BedReader* bedReader,
    const Container::HostVector* outcomes) {

  DataHandler* dataHandler = dataHandlerFactory->constructDataHandler();
  device->setActiveDevice();

  KernelWrapperFactory kernelWrapperFactory;
  StreamFactory streamFactory;
  StatisticsFactory statisticsFactory;
  const Container::DeviceVector& deviceOutcomes = device->getOutcomes();

  Stream* stream = streamFactory.constructStream(*device);
  KernelWrapper* kernelWrapper = kernelWrapperFactory.constructKernelWrapper(*stream);
  HostToDevice hostToDevice(stream->getCudaStream()); //FIXME
  DeviceToHost deviceToHost(stream->getCudaStream()); //FIXME

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
  Model::ModelHandler* modelHandler = new Model::GpuModelHandler(statisticsFactory, dataHandler,
      *logisticRegressionConfiguration, logisticRegression);
  CUDA::handleCudaStatus(cudaGetLastError(),
      "Error with initialisation in GPUWorkerThread " << std::this_thread::get_id() << " : ");

  DataHandlerState dataHandlerState = modelHandler->next();
  while(dataHandlerState != DONE){
    const SNP& snp = modelHandler->getCurrentSNP();
    const EnvironmentFactor& envFactor = modelHandler->getCurrentEnvironmentFactor();

    if(dataHandlerState == SKIP){
      //TODO need to add allele freqs and such
      std::cout << snp << "," << envFactor << std::endl;
    }else{
      Statistics* statistics = modelHandler->calculateModel();

      CUDA::handleCudaStatus(cudaGetLastError(),
          "Error with ModelHandler in GPUWorkerThread " << std::this_thread::get_id() << " : ");

      const Container::SNPVector& snpVector = modelHandler->getSNPVector();

      std::cout << snp << "," << envFactor << "," << *statistics << "," << snpVector << std::endl;

      delete statistics;
    } //else

    dataHandlerState = modelHandler->next();
  }

  delete stream;
  delete kernelWrapper;
  delete modelHandler;

}

} /* namespace CUDA */
} /* namespace CuEira */
