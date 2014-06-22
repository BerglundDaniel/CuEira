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

#ifdef CPU
#include <CpuModelHandler.h>
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
  /*
   FileIO::DataFilesReaderFactory dataFilesReaderFactory;
   FileIO::DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configuration);

   PersonHandler* personHandler = dataFilesReader->readPersonInformation();
   EnvironmentFactorHandler* environmentFactorHandler = dataFilesReader->readEnvironmentFactorInformation(
   *personHandler);

   std::vector<SNP*>* snpInformation = dataFilesReader->readSNPInformation();
   int numberOfSNPs = snpInformation->size();

   Container::HostMatrix* covariates = nullptr;
   if(configuration.covariateFileSpecified()){
   std::pair<Container::HostMatrix*, std::vector<std::string>*>* covPair = dataFilesReader->readCovariates(
   *personHandler);
   covariates = covPair->first;
   delete covPair->second;
   delete covPair;
   }

   FileIO::BedReader bedReader(configuration, *personHandler, numberOfSNPs);
   Task::DataQueue* dataQueue = new Task::DataQueue(snpInformation);
   DataHandler* dataHandler = new DataHandler(configuration.getStatisticModel(), bedReader, *environmentFactorHandler,
   *dataQueue);

   #ifdef CPU
   Model::ModelHandler* modelHandler = new Model::CpuModelHandler();
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

   while(modelHandler->next()){
   Statistics* statistics = modelHandler->calculateModel();
   #ifndef CPU
   CUDA::handleCudaStatus(cudaGetLastError(), "Error with logistic regression: ");
   #endif

   //TODO print stuff

   delete statistics;
   }

   delete dataFilesReader;
   delete personHandler;
   delete environmentFactorHandler;
   delete snpInformation;
   delete modelHandler;
   delete dataQueue;
   delete covariates;

   #ifdef CPU

   #else
   delete deviceOutcomes;
   CUDA::handleCudaStatus(cudaStreamDestroy(cudaStream), "Failed to destroy cudaStream:");
   CUDA::handleCublasStatus(cublasDestroy(cublasHandle), "Failed to destroy cublas handle:");
   #endif
   */
}
