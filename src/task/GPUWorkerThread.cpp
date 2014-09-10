#include "GPUWorkerThread.h"

namespace CUDA {
namespace CuEira {

void GPUWorkerThread(const Configuration* configuration, const Device* device,
    const DataHandlerFactory* dataHandlerFactory, const FileIO::BedReader* bedReader,
    const Container::HostVector* outcomes) {

  DataHandler* dataHandler = dataHandlerFactory->constructDataHandler();
  device->setActive();

  /*

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

     std::cout << snp << "," << envFactor << "," << *statistics << "," << snpVector << std::endl;

     delete statistics;
     } //else

     dataHandlerState = modelHandler->next();
     }

     delete modelHandler;

     */

}

} /* namespace CUDA */
} /* namespace CuEira */
