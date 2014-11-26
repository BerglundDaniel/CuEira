#include "CudaLogisticRegression.h"

namespace CuEira {
namespace Model {
namespace LogisticRegression {
namespace CUDA {

#ifdef PROFILE
boost::chrono::duration<long long, boost::nano> CudaLogisticRegression::timeSpentTotal;
boost::chrono::duration<long long, boost::nano> CudaLogisticRegression::timeSpentGPU;
boost::chrono::duration<long long, boost::nano> CudaLogisticRegression::timeSpentCPU;
boost::chrono::duration<long long, boost::nano> CudaLogisticRegression::timeSpentTransferFromDevice;
boost::chrono::duration<long long, boost::nano> CudaLogisticRegression::timeSpentTransferToDevice;
#endif

CudaLogisticRegression::CudaLogisticRegression(CudaLogisticRegressionConfiguration* lrConfiguration) :
    LogisticRegression(lrConfiguration), hostToDevice(&lrConfiguration->getHostToDevice()), deviceToHost(
        &lrConfiguration->getDeviceToHost()), kernelWrapper(&lrConfiguration->getKernelWrapper()), lrConfiguration(
        lrConfiguration), informationMatrixDevice(&lrConfiguration->getInformationMatrix()), betaCoefficentsDevice(
        &lrConfiguration->getBetaCoefficents()), predictorsDevice(&lrConfiguration->getPredictors()), outcomesDevice(
        &lrConfiguration->getOutcomes()), probabilitesDevice(&lrConfiguration->getProbabilites()), scoresDevice(
        &lrConfiguration->getScores()), workMatrixNxMDevice(&lrConfiguration->getWorkMatrixNxM()), workVectorNx1Device(
        &lrConfiguration->getWorkVectorNx1()), oneVector((*predictorsDevice)(0)), defaultBetaCoefficents(
        &lrConfiguration->getDefaultBetaCoefficents()) {

}

CudaLogisticRegression::CudaLogisticRegression() :
    LogisticRegression(), hostToDevice(nullptr), deviceToHost(nullptr), kernelWrapper(nullptr), informationMatrixDevice(
        nullptr), betaCoefficentsDevice(nullptr), predictorsDevice(nullptr), outcomesDevice(nullptr), probabilitesDevice(
        nullptr), scoresDevice(nullptr), workMatrixNxMDevice(nullptr), workVectorNx1Device(nullptr), lrConfiguration(
        nullptr), oneVector(nullptr), defaultBetaCoefficents(nullptr) {

}

CudaLogisticRegression::~CudaLogisticRegression() {
  delete oneVector;
}

LogisticRegressionResult* CudaLogisticRegression::calculate() {
#ifdef PROFILE
  boost::chrono::system_clock::time_point before = boost::chrono::system_clock::now();
#endif

  PRECISION diffSumHost = 0;
  logLikelihood = 0;

  Container::PinnedHostVector* betaCoefficentsHost = new Container::PinnedHostVector(numberOfPredictors);
  blasWrapper->copyVector(*defaultBetaCoefficents, *betaCoefficentsHost);

#ifdef PROFILE
  boost::chrono::system_clock::time_point beforeTransfer_beta = boost::chrono::system_clock::now();
#endif
  hostToDevice->transferVector(*defaultBetaCoefficents, betaCoefficentsDevice->getMemoryPointer());
  kernelWrapper->syncStream();
#ifdef PROFILE
  boost::chrono::system_clock::time_point afterTransfer_beta = boost::chrono::system_clock::now();
  timeSpentTransferToDevice+=afterTransfer_beta - beforeTransfer_beta;
#endif

  Container::PinnedHostMatrix* informationMatrixHost = new Container::PinnedHostMatrix(numberOfPredictors,
      numberOfPredictors);
  Container::RegularHostMatrix* inverseInformationMatrixHost = new Container::RegularHostMatrix(numberOfPredictors,
      numberOfPredictors);

  int iterationNumber = 1;
  for(iterationNumber = 1; iterationNumber < maxIterations; ++iterationNumber){
#ifdef PROFILE
    boost::chrono::system_clock::time_point beforeGPU = boost::chrono::system_clock::now();
#endif

    calcuateProbabilites(*predictorsDevice, *betaCoefficentsDevice, *probabilitesDevice, *workVectorNx1Device);

    calculateScores(*predictorsDevice, *outcomesDevice, *probabilitesDevice, *scoresDevice, *workVectorNx1Device);

    calculateInformationMatrix(*predictorsDevice, *probabilitesDevice, *workVectorNx1Device, *informationMatrixDevice,
        *workMatrixNxMDevice);

#ifdef PROFILE
    kernelWrapper->syncStream();

    boost::chrono::system_clock::time_point afterGPU = boost::chrono::system_clock::now();
    timeSpentGPU+=afterGPU - beforeGPU;

    boost::chrono::system_clock::time_point beforeTransfer = boost::chrono::system_clock::now();
#endif

    //Transfer needed data to host
    deviceToHost->transferMatrix(*informationMatrixDevice, informationMatrixHost->getMemoryPointer());
#ifdef FERMI
    kernelWrapper->syncStream();
#endif

    deviceToHost->transferVector(*scoresDevice, scoresHost->getMemoryPointer());
    kernelWrapper->syncStream(); //Because of the transfers

#ifdef PROFILE
    boost::chrono::system_clock::time_point afterTransfer = boost::chrono::system_clock::now();
    timeSpentTransferFromDevice+=afterTransfer - beforeTransfer;

    boost::chrono::system_clock::time_point beforeCPU = boost::chrono::system_clock::now();
#endif

    //Copy beta to old beta
    blasWrapper->copyVector(*betaCoefficentsHost, *betaCoefficentsOldHost);

    invertInformationMatrix(*informationMatrixHost, *inverseInformationMatrixHost, *uSVD, *sigma, *vtSVD,
        *workMatrixMxMHost);

    calculateNewBeta(*inverseInformationMatrixHost, *scoresHost, *betaCoefficentsHost);

    calculateDifference(*betaCoefficentsHost, *betaCoefficentsOldHost, diffSumHost);

#ifdef PROFILE
    boost::chrono::system_clock::time_point afterCPU = boost::chrono::system_clock::now();
    timeSpentCPU+=afterCPU - beforeCPU;
#endif

    if(diffSumHost < convergenceThreshold){
#ifdef PROFILE
      boost::chrono::system_clock::time_point beforeGPU_likeli = boost::chrono::system_clock::now();
#endif
      calculateLogLikelihood(*outcomesDevice, *oneVector, *probabilitesDevice, *workVectorNx1Device, logLikelihood);

#ifdef PROFILE
      kernelWrapper->syncStream();
      boost::chrono::system_clock::time_point afterGPU_likeli = boost::chrono::system_clock::now();
      timeSpentGPU+=afterGPU_likeli - beforeGPU_likeli;

      boost::chrono::system_clock::time_point beforeTransfer_likeli = boost::chrono::system_clock::now();
#endif

      //Transfer the information matrix again since it was destroyed during the SVD.
      deviceToHost->transferMatrix(*informationMatrixDevice, informationMatrixHost->getMemoryPointer());

#ifdef PROFILE
      kernelWrapper->syncStream();
      boost::chrono::system_clock::time_point afterTransfer_likeli = boost::chrono::system_clock::now();
      timeSpentTransferFromDevice+=afterTransfer_likeli - beforeTransfer_likeli;
#endif

      break;
    }else{
#ifdef PROFILE
      boost::chrono::system_clock::time_point beforeTransfer_new_it = boost::chrono::system_clock::now();
#endif
      hostToDevice->transferVector(*betaCoefficentsHost, betaCoefficentsDevice->getMemoryPointer());
      kernelWrapper->syncStream();
#ifdef PROFILE
      boost::chrono::system_clock::time_point afterTransfer_new_it = boost::chrono::system_clock::now();
      timeSpentTransferToDevice+=afterTransfer_new_it - beforeTransfer_new_it;
#endif
    }
  } /* for iterationNumber */

  kernelWrapper->syncStream();

#ifdef PROFILE
  boost::chrono::system_clock::time_point after = boost::chrono::system_clock::now();
  timeSpentTotal+=after - before;
#endif

  return new LogisticRegressionResult(betaCoefficentsHost, informationMatrixHost, inverseInformationMatrixHost,
      iterationNumber, logLikelihood);
}
void CudaLogisticRegression::calcuateProbabilites(const DeviceMatrix& predictorsDevice,
    const DeviceVector& betaCoefficentsDevice, DeviceVector& probabilitesDevice, DeviceVector& workVectorNx1Device) {
  kernelWrapper->matrixVectorMultiply(predictorsDevice, betaCoefficentsDevice, workVectorNx1Device);
  kernelWrapper->logisticTransform(workVectorNx1Device, probabilitesDevice);
}

void CudaLogisticRegression::calculateScores(const DeviceMatrix& predictorsDevice, const DeviceVector& outcomesDevice,
    const DeviceVector& probabilitesDevice, DeviceVector& scoresDevice, DeviceVector& workVectorNx1Device) {
  kernelWrapper->elementWiseDifference(outcomesDevice, probabilitesDevice, workVectorNx1Device);
  kernelWrapper->matrixTransVectorMultiply(predictorsDevice, workVectorNx1Device, scoresDevice);
}

void CudaLogisticRegression::calculateInformationMatrix(const DeviceMatrix& predictorsDevice,
    const DeviceVector& probabilitesDevice, DeviceVector& workVectorNx1Device, DeviceMatrix& informationMatrixDevice,
    DeviceMatrix& workMatrixNxMDevice) {
  kernelWrapper->probabilitesMultiplyProbabilites(probabilitesDevice, workVectorNx1Device);
  kernelWrapper->columnByColumnMatrixVectorElementWiseMultiply(predictorsDevice, workVectorNx1Device,
      workMatrixNxMDevice);
  kernelWrapper->matrixTransMatrixMultiply(predictorsDevice, workMatrixNxMDevice, informationMatrixDevice);
}

void CudaLogisticRegression::calculateLogLikelihood(const DeviceVector& outcomesDevice, const DeviceVector& oneVector,
    const DeviceVector& probabilitesDevice, DeviceVector& workVectorNx1Device, PRECISION& logLikelihood) {
  kernelWrapper->logLikelihoodParts(outcomesDevice, probabilitesDevice, workVectorNx1Device);
  kernelWrapper->sumResultToHost(workVectorNx1Device, oneVector, logLikelihood);
}

} /* namespace CUDA */
} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
