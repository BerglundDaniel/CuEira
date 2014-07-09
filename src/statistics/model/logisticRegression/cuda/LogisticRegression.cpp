#include "LogisticRegression.h"

namespace CuEira {
namespace Model {
namespace LogisticRegression {

LogisticRegression::LogisticRegression(LogisticRegressionConfiguration* lrConfiguration,
    const HostToDevice& hostToDevice, const DeviceToHost& deviceToHost) :
    hostToDevice(hostToDevice), deviceToHost(deviceToHost), kernelWrapper(lrConfiguration->getKernelWrapper()), lrConfiguration(
        lrConfiguration), maxIterations(lrConfiguration->getNumberOfMaxIterations()), convergenceThreshold(
        lrConfiguration->getConvergenceThreshold()), numberOfRows(lrConfiguration->getNumberOfRows()), numberOfPredictors(
        lrConfiguration->getNumberOfPredictors()), informationMatrixDevice(lrConfiguration->getInformationMatrix()), betaCoefficentsDevice(
        lrConfiguration->getBetaCoefficents()), scoresHost(new PinnedHostVector(numberOfPredictors)), logLikelihood(
        new PRECISION(0)), mklWrapper(), betaCoefficentsOldHost(new Container::PinnedHostVector(numberOfPredictors)), predictorsDevice(
        lrConfiguration->getPredictors()), outcomesDevice(lrConfiguration->getOutcomes()), probabilitesDevice(
        lrConfiguration->getProbabilites()), scoresDevice(lrConfiguration->getScores()), workMatrixNxMDevice(
        lrConfiguration->getWorkMatrixNxM()), workVectorNx1Device(lrConfiguration->getWorkVectorNx1()), sigma(
        new PinnedHostVector(numberOfPredictors)), uSVD(new PinnedHostMatrix(numberOfPredictors, numberOfPredictors)), vtSVD(
        new PinnedHostMatrix(numberOfPredictors, numberOfPredictors)), workMatrixMxMHost(
        new PinnedHostMatrix(numberOfPredictors, numberOfPredictors)), oneVector(predictorsDevice(0)) {

}

LogisticRegression::~LogisticRegression() {
  delete lrConfiguration;
  delete betaCoefficentsOldHost;
  delete scoresHost;
  delete sigma;
  delete uSVD;
  delete vtSVD;
  delete workMatrixMxMHost;
  delete oneVector;
}

LogisticRegressionResult* LogisticRegression::calculate() {
  PRECISION* diffSumHost = new PRECISION(0);

  //Somethings are initialised here since the result wrapper will take responsibility for them at the end so can't reuse them
  Container::HostVector* betaCoefficentsHost = deviceToHost.transferVector(&betaCoefficentsDevice);
  Container::HostMatrix* informationMatrixHost = new Container::PinnedHostMatrix(numberOfPredictors,
      numberOfPredictors);
  Container::HostMatrix* inverseInformationMatrixHost = new Container::PinnedHostMatrix(numberOfPredictors,
      numberOfPredictors);

  for(iterationNumber = 0; iterationNumber < maxIterations; ++iterationNumber){
    calcuateProbabilites(predictorsDevice, betaCoefficentsDevice, probabilitesDevice, workVectorNx1Device);

    calculateScores(predictorsDevice, outcomesDevice, probabilitesDevice, scoresDevice, workVectorNx1Device);

    calculateInformationMatrix(predictorsDevice, probabilitesDevice, workVectorNx1Device, informationMatrixDevice,
        workMatrixNxMDevice);

    //Copy beta to old beta
    mklWrapper.copyVector(*betaCoefficentsHost, *betaCoefficentsOldHost);

    //Transfer needed data to host
    deviceToHost.transferMatrix(&informationMatrixDevice, informationMatrixHost->getMemoryPointer());
    deviceToHost.transferVector(&scoresDevice, scoresHost->getMemoryPointer());
    kernelWrapper.syncStream();

    invertInformationMatrix(*informationMatrixHost, *inverseInformationMatrixHost, *uSVD, *sigma, *vtSVD,
        *workMatrixMxMHost);

    calculateNewBeta(*inverseInformationMatrixHost, *scoresHost, *betaCoefficentsHost);

    calculateDifference(*betaCoefficentsHost, *betaCoefficentsOldHost, diffSumHost);

    if(*diffSumHost < convergenceThreshold){
      calculateLogLikelihood(outcomesDevice, *oneVector, probabilitesDevice, workVectorNx1Device, logLikelihood);

      //Transfer the information matrix again since it was destroyed during the SVD.
      deviceToHost.transferMatrix(&informationMatrixDevice, informationMatrixHost->getMemoryPointer());

      break;
    }else{
      hostToDevice.transferVector(betaCoefficentsHost, betaCoefficentsDevice.getMemoryPointer());
      kernelWrapper.syncStream();
    }
  } /* for iterationNumber */

  delete diffSumHost;

  kernelWrapper.syncStream();
  return new LogisticRegressionResult(betaCoefficentsHost, informationMatrixHost, inverseInformationMatrixHost,
      iterationNumber, *logLikelihood);
}
void LogisticRegression::calcuateProbabilites(const DeviceMatrix& predictorsDevice,
    const DeviceVector& betaCoefficentsDevice, DeviceVector& probabilitesDevice, DeviceVector& workVectorNx1Device) {
  kernelWrapper.matrixVectorMultiply(predictorsDevice, betaCoefficentsDevice, workVectorNx1Device);
  kernelWrapper.logisticTransform(workVectorNx1Device, probabilitesDevice);
}

void LogisticRegression::calculateScores(const DeviceMatrix& predictorsDevice, const DeviceVector& outcomesDevice,
    const DeviceVector& probabilitesDevice, DeviceVector& scoresDevice, DeviceVector& workVectorNx1Device) {
  kernelWrapper.elementWiseDifference(outcomesDevice, probabilitesDevice, workVectorNx1Device);
  kernelWrapper.matrixTransVectorMultiply(predictorsDevice, workVectorNx1Device, scoresDevice);
}

void LogisticRegression::calculateInformationMatrix(const DeviceMatrix& predictorsDevice,
    const DeviceVector& probabilitesDevice, DeviceVector& workVectorNx1Device, DeviceMatrix& informationMatrixDevice,
    DeviceMatrix& workMatrixNxMDevice) {
  kernelWrapper.probabilitesMultiplyProbabilites(probabilitesDevice, workVectorNx1Device);
  kernelWrapper.columnByColumnMatrixVectorElementWiseMultiply(predictorsDevice, workVectorNx1Device,
      workMatrixNxMDevice);
  kernelWrapper.matrixTransMatrixMultiply(predictorsDevice, workMatrixNxMDevice, informationMatrixDevice);
}

void LogisticRegression::invertInformationMatrix(HostMatrix& informationMatrixHost,
    HostMatrix& inverseInformationMatrixHost, HostMatrix& uSVD, HostVector& sigma, HostMatrix& vtSVD,
    HostMatrix& workMatrixMxMHost) {
  int size = informationMatrixHost.getNumberOfRows();

  mklWrapper.svd(informationMatrixHost, uSVD, sigma, vtSVD);

  //diag(sigma)*uSVD'
  for(int i = 0; i < size; ++i){
    PRECISION inverseSigma;
    if(fabs(sigma(i)) > 1e-5){
      inverseSigma = 1 / sigma(i);
    }else{
      inverseSigma = 0;
    }
    //cblas_ FIXME use scal
    for(int k = 0; k < size; ++k){
      workMatrixMxMHost(i, k) = inverseSigma * uSVD(k, i);
    }
  }

  mklWrapper.matrixTransMatrixMultiply(vtSVD, workMatrixMxMHost, inverseInformationMatrixHost, 1, 0);
}

void LogisticRegression::calculateNewBeta(HostMatrix& inverseInformationMatrixHost, HostVector& scoresHost,
    HostVector& betaCoefficentsHost) {
  //beta=inv*scores+beta
  mklWrapper.matrixVectorMultiply(inverseInformationMatrixHost, scoresHost, betaCoefficentsHost, 1, 1);
}

void LogisticRegression::calculateDifference(const HostVector& betaCoefficentsHost, HostVector& betaCoefficentsOldHost,
    PRECISION* diffSumHost) {
  mklWrapper.differenceElememtWise(betaCoefficentsHost, betaCoefficentsOldHost);
  mklWrapper.absoluteSum(betaCoefficentsOldHost, diffSumHost);
}

void LogisticRegression::calculateLogLikelihood(const DeviceVector& outcomesDevice, const DeviceVector& oneVector,
    const DeviceVector& probabilitesDevice, DeviceVector& workVectorNx1Device, PRECISION* logLikelihood) {
  kernelWrapper.logLikelihoodParts(outcomesDevice, probabilitesDevice, workVectorNx1Device);
  kernelWrapper.sumResultToHost(workVectorNx1Device, oneVector, logLikelihood);
}

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
