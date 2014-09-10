#include <iostream>
#include <stdexcept>
#include <vector>
#include <thread>

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
#include <DataHandlerFactory.h>

#ifdef CPU
//#include <CpuModelHandler.h>
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <HostToDevice.h>
#include <DeviceToHost.h>
#include <CudaAdapter.cu>
#include <GpuModelHandler.h>
#include <LogisticRegressionConfiguration.h>
#include <CudaException.h>
#include <GPUWorkerThread.h>
#include <Device.h>
#endif

/**
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
int main(int argc, char* argv[]) {
  using namespace CuEira;
  std::cerr << "Starting" << std::endl;
#ifdef DEBUG
  std::cerr << "WARNING CuEira was compiled in debug mode, this can affect performance." << std::endl;
#endif

  Configuration configuration(argc, argv);
  FileIO::DataFilesReaderFactory dataFilesReaderFactory;
  Container::SNPVectorFactory snpVectorFactory(configuration);
  StatisticsFactory statisticsFactory;
  AlleleStatisticsFactory alleleStatisticsFactory;

  FileIO::DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configuration);

  PersonHandler* personHandler = dataFilesReader->readPersonInformation();
  EnvironmentFactorHandler* environmentFactorHandler = dataFilesReader->readEnvironmentFactorInformation(
      *personHandler);
  std::vector<SNP*>* snpInformation = dataFilesReader->readSNPInformation();

  const int numberOfSNPs = snpInformation->size();
  const int numberOfIndividualsToInclude = personHandler->getNumberOfIndividualsToInclude();
  const Container::HostVector& outcomes = personHandler->getOutcomes();

  ContingencyTableFactory contingencyTableFactory(outcomes);
  DataHandlerFactory dataHandlerFactory(configuration, contingencyTableFactory, bedReader, *environmentFactorHandler,
      *dataQueue);

  FileIO::BedReader* bedReader = new FileIO::BedReader(configuration, snpVectorFactory, alleleStatisticsFactory,
      *personHandler, numberOfSNPs);
  Task::DataQueue* dataQueue = new Task::DataQueue(snpInformation);

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

#ifdef CPU
  //TODO
  //Model::ModelHandler* modelHandler = new Model::CpuModelHandler();
#else
  //GPU
  const int numberOfDevices;
  const int numberOfStreams = configuration.getNumberOfStreams();
  const int numberOfThreads = numberOfDevices * numberOfStreams;
  cudaGetDeviceCount(&numberOfDevices);

  if(numberOfDevices == 0){
    throw new CudaException("No cuda devices found.");
  }
  std::cerr << "Calculating using " << numberOfDevices << " with " << numberOfStreams << " each." << std::endl;

  std::vector<CUDA::Device*> devices(numberOfDevices);
  std::vector<std::thread*> workers(numberOfThreads);

  for(int deviceNumber = 0; deviceNumber < numberOfDevices; ++deviceNumber){
    CUDA::Device* device = new CUDA::Device(deviceNumber);
    devices[deviceNumber] = device;

    //Start threads
    for(int streamNumber = 0; streamNumber < numberOfStreams; ++streamNumber){
      //TODO fix covariates
      std::thread* thread = new std::thread(GPUWorkerThread, &configuration, device, dataHandlerFactory, bedReader,
          outcomes);
    }
  }

  for(int threadNumber = 0; threadNumber < numberOfThreads; ++threadNumber){
    workers[threadNumber]->join();
  }

  for(int threadNumber = 0; threadNumber < numberOfThreads; ++threadNumber){
    delete workers[threadNumber];
  }

  for(int deviceNumber = 0; deviceNumber < numberOfDevices; ++deviceNumber){
    delete devices[deviceNumber];
  }

#endif

  delete bedReader;
  delete dataFilesReader;
  delete personHandler;
  delete environmentFactorHandler;
  delete dataQueue;
  delete covariates;
  delete covariatesNames;

  std::cerr << "Done" << std::endl;

}
