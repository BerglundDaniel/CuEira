#include <iostream>
#include <stdexcept>
#include <vector>

#include <Configuration.h>
#include <DataFilesReaderFactory.h>
#include <DataFilesReader.h>
#include <SNPVector.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <PersonHandler.h>
#include <EnvironmentFactor.h>
#include <Statistics.h>
#include <Recode.h>
#include <DataHandler.h>
#include <ModelHandler.h>
#include <EnvironmentVector.h>
#include <InteractionVector.h>

#ifdef CPU
//#include <CpuModelHandler.h>
#else
#include <HostToDevice.h>
#include <DeviceToHost.h>
#include <CudaAdapter.cu>
#include <GpuModelHandler.h>
#include <LogisticRegressionConfiguration.h>
#endif

/**
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
int main(int argc, char* argv[]) {
  using namespace CuEira;

  Configuration configuration(argc, argv);

  FileIO::DataFilesReaderFactory dataFilesReaderFactory;
  FileIO::DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configuration);

  PersonHandler* personHandler = dataFilesReader->readPersonInformation();
  EnvironmentFactorHandler* environmentFactorHandler = dataFilesReader->readEnvironmentFactorInformation(
      *personHandler);

  std::vector<SNP*>* snpInformation = dataFilesReader->readSNPInformation();
  const int numberOfSNPs = snpInformation->size();
  const int numberOfIndividualsToInclude = environmentFactorHandler->getNumberOfIndividualsToInclude();

  Container::HostMatrix* covariates = nullptr;
  std::vector<std::string>* covariatesNames = nullptr;
  int numberOfCovariates = 0;
  if(configuration.covariateFileSpecified()){
    std::pair<Container::HostMatrix*, std::vector<std::string>*>* covPair = dataFilesReader->readCovariates(
        *personHandler);
    covariates = covPair->first;
    covariatesNames = covPair->second;
    delete covPair;

    numberOfCovariates = covariates->getNumberOfColumns();
  }

  Container::SNPVectorFactory snpVectorFactory(configuration, numberOfIndividualsToInclude);

  FileIO::BedReader bedReader(configuration, snpVectorFactory, *personHandler, numberOfSNPs);
  Task::DataQueue* dataQueue = new Task::DataQueue(snpInformation);

  //FIXME this part to factory for DataHandler
  Container::EnvironmentVector* environmentVector = new Container::EnvironmentVector(*environmentFactorHandler);
  Container::InteractionVector* interactionVector = new Container::InteractionVector(*environmentVector);
  DataHandler* dataHandler = new DataHandler(configuration.getStatisticModel(), bedReader, environmentFactorHandler->getHeaders(),
      *dataQueue, environmentVector, interactionVector);

#ifdef CPU
  //Model::ModelHandler* modelHandler = new Model::CpuModelHandler();
  Model::ModelHandler* modelHandler=nullptr;
#else
  //GPU
  cudaStream_t cudaStream;
  cublasHandle_t cublasHandle;
  CUDA::handleCublasStatus(cublasCreate(&cublasHandle), "Failed to create cublas handle:");
  CUDA::handleCudaStatus(cudaStreamCreate(&cudaStream), "Failed to create cudaStream:");

  CUDA::HostToDevice hostToDevice(cudaStream);
  CUDA::DeviceToHost deviceToHost(cudaStream);
  Container::DeviceVector* deviceOutcomes = hostToDevice.transferVector(&personHandler->getOutcomes());

  CUDA::KernelWrapper kernelWrapper(cudaStream, cublasHandle);

  Model::LogisticRegression::LogisticRegressionConfiguration* logisticRegressionConfiguration;

  if(configuration.covariateFileSpecified()){
    logisticRegressionConfiguration = new Model::LogisticRegression::LogisticRegressionConfiguration(configuration,
        hostToDevice, *deviceOutcomes, kernelWrapper);
  }else{
    logisticRegressionConfiguration = new Model::LogisticRegression::LogisticRegressionConfiguration(configuration,
        hostToDevice, *deviceOutcomes, kernelWrapper, *covariates);
  }

  Model::ModelHandler* modelHandler = new Model::GpuModelHandler(dataHandler, logisticRegressionConfiguration,
      hostToDevice, deviceToHost);

  CUDA::handleCudaStatus(cudaGetLastError(), "Error with initialisation in main: ");
#endif

  std::cout << "header"; //FIXME

  for(int i = 0; i < numberOfCovariates; ++i){
    std::cout << (*covariatesNames)[i] << "_OR, " << (*covariatesNames)[i] << "_OR_L, " << (*covariatesNames)[i]
        << "_OR_H";
  }
  std::cout << std::endl;

  while(modelHandler->next()){
    Statistics* statistics = modelHandler->calculateModel();
    const Container::SNPVector& snpVector = modelHandler->getSNPVector();
    const SNP& snp = modelHandler->getCurrentSNP();
    const EnvironmentFactor& envFactor = modelHandler->getCurrentEnvironmentFactor();

    std::cout << snp << ", " << envFactor << ", " << *statistics << ", " << snpVector << std::endl;

    delete statistics;
  }

  delete dataFilesReader;
  delete personHandler;
  delete environmentFactorHandler;
  delete snpInformation; //FIXME already in task, need to delete snps somehow
  delete modelHandler;
  delete dataQueue;
  delete covariates;
  delete covariatesNames;

#ifdef CPU

#else
  delete deviceOutcomes;
  CUDA::handleCudaStatus(cudaStreamDestroy(cudaStream), "Failed to destroy cudaStream:");
  CUDA::handleCublasStatus(cublasDestroy(cublasHandle), "Failed to destroy cublas handle:");
#endif

}
