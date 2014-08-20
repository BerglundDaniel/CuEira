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

  Configuration configuration(argc, argv);
  std::cerr << "m1" << std::endl;
  FileIO::DataFilesReaderFactory dataFilesReaderFactory;
  FileIO::DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configuration);
  std::cerr << "m2" << std::endl;
  PersonHandler* personHandler = dataFilesReader->readPersonInformation();
  std::cerr << "m2_1" << std::endl;
  EnvironmentFactorHandler* environmentFactorHandler = dataFilesReader->readEnvironmentFactorInformation(
      *personHandler);
  std::cerr << "m3" << std::endl;
  std::vector<SNP*>* snpInformation = dataFilesReader->readSNPInformation();
  const int numberOfSNPs = snpInformation->size();
  const int numberOfIndividualsToInclude = environmentFactorHandler->getNumberOfIndividualsToInclude();
  std::cerr << "m4" << std::endl;
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
  std::cerr << "m5" << std::endl;
  Container::SNPVectorFactory snpVectorFactory(configuration, numberOfIndividualsToInclude);
  StatisticsFactory statisticsFactory;
  std::cerr << "m6" << std::endl;
  FileIO::BedReader bedReader(configuration, snpVectorFactory, *personHandler, numberOfSNPs);

  Task::DataQueue* dataQueue = new Task::DataQueue(*snpInformation);
  std::cerr << "m7" << std::endl;
  //FIXME this part to factory for DataHandler
  Container::EnvironmentVector* environmentVector = new Container::EnvironmentVector(*environmentFactorHandler);
  Container::InteractionVector* interactionVector = new Container::InteractionVector(*environmentVector);
  DataHandler* dataHandler = new DataHandler(configuration.getStatisticModel(), bedReader,
      environmentFactorHandler->getHeaders(), *dataQueue, environmentVector, interactionVector);

#ifdef CPU
  //Model::ModelHandler* modelHandler = new Model::CpuModelHandler();
  Model::ModelHandler* modelHandler=nullptr;
#else
  //GPU
  std::cerr << "m8" << std::endl;
  cudaStream_t cudaStream;
  cublasHandle_t cublasHandle;
  CUDA::handleCublasStatus(cublasCreate(&cublasHandle), "Failed to create cublas handle:");
  CUDA::handleCudaStatus(cudaStreamCreate(&cudaStream), "Failed to create cudaStream:");
  CUDA::handleCublasStatus(cublasSetStream(cublasHandle, cudaStream), "Failed to set cuda stream:");

  std::cerr << "m9" << std::endl;
  CUDA::HostToDevice hostToDevice(cudaStream);
  CUDA::DeviceToHost deviceToHost(cudaStream);
  Container::DeviceVector* deviceOutcomes = hostToDevice.transferVector(&(personHandler->getOutcomes()));
  CUDA::handleCudaStatus(cudaGetLastError(), "Error transferring outcomes to device in main: ");
  std::cerr << "m10" << std::endl;
  CUDA::KernelWrapper kernelWrapper(cudaStream, cublasHandle);

  Model::LogisticRegression::LogisticRegressionConfiguration* logisticRegressionConfiguration = nullptr;
  std::cerr << "m11" << std::endl;
  if(configuration.covariateFileSpecified()){
    logisticRegressionConfiguration = new Model::LogisticRegression::LogisticRegressionConfiguration(configuration,
        hostToDevice, *deviceOutcomes, kernelWrapper, *covariates);
  }else{
    logisticRegressionConfiguration = new Model::LogisticRegression::LogisticRegressionConfiguration(configuration,
        hostToDevice, *deviceOutcomes, kernelWrapper);
  }
  std::cerr << "m12" << std::endl;
  Model::LogisticRegression::LogisticRegression* logisticRegression = new Model::LogisticRegression::LogisticRegression(
      logisticRegressionConfiguration, hostToDevice, deviceToHost);
  Model::ModelHandler* modelHandler = new Model::GpuModelHandler(statisticsFactory, dataHandler,
      *logisticRegressionConfiguration, logisticRegression);

  CUDA::handleCudaStatus(cudaGetLastError(), "Error with initialisation in main: ");
#endif
  std::cerr << "m13" << std::endl;
  std::cout
      << "snp_id,risk_allele,minor,major,env_id,ap,reri,OR_snp,OR_snp_L,OR_snp_H,OR_env,OR_env_L,OR_env_H,OR_inter,OR_inter_L,OR_inter_H,";

  for(int i = 0; i < numberOfCovariates; ++i){
    std::cout << (*covariatesNames)[i] << "_cov_OR," << (*covariatesNames)[i] << "_cov_OR_L," << (*covariatesNames)[i]
        << "_cov_OR_H,";
  }

  std::cout << "recode"; //FIXME
  std::cout << std::endl;

  const Container::HostVector& outcomes = personHandler->getOutcomes(); //TMP

  while(modelHandler->next()){
    Statistics* statistics = modelHandler->calculateModel();
    CUDA::handleCudaStatus(cudaGetLastError(), "Error with ModelHandler in Main: ");

    const Container::SNPVector& snpVector = modelHandler->getSNPVector();
    const SNP& snp = modelHandler->getCurrentSNP();
    const EnvironmentFactor& envFactor = modelHandler->getCurrentEnvironmentFactor();

    const Container::EnvironmentVector& envVector = modelHandler->getEnvironmentVector();
    const Container::InteractionVector& interVector = modelHandler->getInteractionVector();
    const Container::HostVector& snpData = snpVector.getRecodedData();
    const Container::HostVector& envData = envVector.getRecodedData();
    const Container::HostVector& interData = interVector.getRecodedData();

    std::cerr << std::endl;
    std::cerr << "snp " << snp.getId().getString() << std::endl;
    for(int i = 0; i < numberOfSNPs; ++i){
      std::cerr << outcomes(i) << "," << snpData(i) << "," << envData(i) << "," << interData(i) << std::endl;
    }
    std::cerr << std::endl;

    std::cout << snp << "," << envFactor << "," << *statistics << "," << snpVector << std::endl;

    delete statistics;
  }

  std::cerr << "delete stuff" << std::endl;

  delete dataFilesReader;
  delete personHandler;
  delete environmentFactorHandler;
  delete modelHandler;
  delete dataQueue;
  delete covariates;
  delete covariatesNames;

  for(int i = 0; i < numberOfSNPs; ++i){
    delete (*snpInformation)[i];
  }
  delete snpInformation;

#ifdef CPU

#else
  delete deviceOutcomes;
  CUDA::handleCudaStatus(cudaStreamDestroy(cudaStream), "Failed to destroy cudaStream:");
  CUDA::handleCublasStatus(cublasDestroy(cublasHandle), "Failed to destroy cublas handle:");
#endif

  std::cerr << "end" << std::endl;

}
