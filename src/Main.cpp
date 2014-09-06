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
#include <StatisticsFactory.h>
#include <DataHandlerState.h>
#include <ContingencyTableFactory.h>
#include <AlleleStatisticsFactory.h>

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
  std::cerr << "Starting" << std::endl;
#ifdef DEBUG
  std::cerr << "CuEira was compiled in debug mode, this can affect performance." << std::endl;
#endif

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

  Container::SNPVectorFactory snpVectorFactory(configuration);
  StatisticsFactory statisticsFactory;
  AlleleStatisticsFactory alleleStatisticsFactory;
  ContingencyTableFactory contingencyTableFactory(personHandler->getOutcomes());
  DataHandlerFactory dataHandlerFactory(configuration, contingencyTableFactory);

  FileIO::BedReader bedReader(configuration, snpVectorFactory, alleleStatisticsFactory, *personHandler, numberOfSNPs);

  Task::DataQueue* dataQueue = new Task::DataQueue(snpInformation);

  DataHandler* dataHandler = dataHandlerFactory.constructDataHandler(bedReader, environmentFactorHandler, *dataQueue);

#ifdef CPU
  //Model::ModelHandler* modelHandler = new Model::CpuModelHandler();
  Model::ModelHandler* modelHandler=nullptr;
#else
  //GPU
  cudaStream_t cudaStream;
  cublasHandle_t cublasHandle;
  CUDA::handleCublasStatus(cublasCreate(&cublasHandle), "Failed to create cublas handle:");
  CUDA::handleCudaStatus(cudaStreamCreate(&cudaStream), "Failed to create cudaStream:");
  CUDA::handleCublasStatus(cublasSetStream(cublasHandle, cudaStream), "Failed to set cuda stream:");

  CUDA::HostToDevice hostToDevice(cudaStream);
  CUDA::DeviceToHost deviceToHost(cudaStream);
  Container::DeviceVector* deviceOutcomes = hostToDevice.transferVector(&(personHandler->getOutcomes()));
  CUDA::handleCudaStatus(cudaGetLastError(), "Error transferring outcomes to device in main: ");
  CUDA::KernelWrapper kernelWrapper(cudaStream, cublasHandle);

  Model::LogisticRegression::LogisticRegressionConfiguration* logisticRegressionConfiguration = nullptr;

  if(configuration.covariateFileSpecified()){
    logisticRegressionConfiguration = new Model::LogisticRegression::LogisticRegressionConfiguration(configuration,
        hostToDevice, *deviceOutcomes, kernelWrapper, *covariates);
  }else{
    logisticRegressionConfiguration = new Model::LogisticRegression::LogisticRegressionConfiguration(configuration,
        hostToDevice, *deviceOutcomes, kernelWrapper);
  }

  Model::LogisticRegression::LogisticRegression* logisticRegression = new Model::LogisticRegression::LogisticRegression(
      logisticRegressionConfiguration, hostToDevice, deviceToHost);
  Model::ModelHandler* modelHandler = new Model::GpuModelHandler(statisticsFactory, dataHandler,
      *logisticRegressionConfiguration, logisticRegression);

  CUDA::handleCudaStatus(cudaGetLastError(), "Error with initialisation in main: ");
#endif

  std::cout
      << "snp_id,risk_allele,minor,major,env_id,ap,reri,OR_snp,OR_snp_L,OR_snp_H,OR_env,OR_env_L,OR_env_H,OR_inter,OR_inter_L,OR_inter_H,";

  for(int i = 0; i < numberOfCovariates; ++i){
    std::cout << (*covariatesNames)[i] << "_cov_OR," << (*covariatesNames)[i] << "_cov_OR_L," << (*covariatesNames)[i]
        << "_cov_OR_H,";
  }

  std::cout << "recode"; //FIXME
  std::cout << std::endl;

  const Container::HostVector& outcomes = personHandler->getOutcomes(); //TMP

  DataHandlerState dataHandlerState = modelHandler->next();
  while(dataHandlerState != DONE){

    const SNP& snp = modelHandler->getCurrentSNP();
    const EnvironmentFactor& envFactor = modelHandler->getCurrentEnvironmentFactor();

    if(dataHandlerState == SKIP){
      //TODO need to add allele freqs and such
      std::cout << snp << "," << envFactor << std::endl;
    }else{
      Statistics* statistics = modelHandler->calculateModel();
#ifndef CPU
      CUDA::handleCudaStatus(cudaGetLastError(), "Error with ModelHandler in Main: ");
#endif

      const Container::SNPVector& snpVector = modelHandler->getSNPVector();
      const Container::EnvironmentVector& envVector = modelHandler->getEnvironmentVector();
      const Container::InteractionVector& interVector = modelHandler->getInteractionVector();
      const Container::HostVector& snpData = snpVector.getRecodedData();
      const Container::HostVector& envData = envVector.getRecodedData();
      const Container::HostVector& interData = interVector.getRecodedData();
      /*
       std::cerr << std::endl;
       std::cerr << "snp " << snp.getId().getString() << std::endl;
       for(int i = 0; i < numberOfIndividualsToInclude; ++i){
       std::cerr << outcomes(i) << "," << snpData(i) << "," << envData(i) << "," << interData(i) << std::endl;
       }
       std::cerr << std::endl;
       */
      std::cout << snp << "," << envFactor << "," << *statistics << "," << snpVector << std::endl;

      delete statistics;
    } //else

    dataHandlerState = modelHandler->next();
  }

  std::cerr << "delete stuff" << std::endl;

  delete dataFilesReader;
  delete personHandler;
  delete environmentFactorHandler;
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

  std::cerr << "end" << std::endl;

}
