#include "LogisticRegression.h"

namespace CuEira {
namespace Model {
namespace LogisticRegression {

LogisticRegression::LogisticRegression(LogisticRegressionConfiguration& lrConfiguration,
    const HostToDevice& hostToDevice, const DeviceToHost& deviceToHost) :
    hostToDevice(hostToDevice), deviceToHost(deviceToHost), kernelWrapper(lrConfiguration.getKernelWrapper()), lrConfiguration(
        lrConfiguration), maxIterations(lrConfiguration.getNumberOfMaxIterations()), convergenceThreshold(
        lrConfiguration.getConvergenceThreshold()), numberOfRows(lrConfiguration.getNumberOfRows()), numberOfPredictors(
        lrConfiguration.getNumberOfPredictors()), informationMatrixDevice(lrConfiguration.getInformationMatrix()), betaCoefficentsDevice(
        lrConfiguration.getBetaCoefficents()), scoresHost(new PinnedHostVector(numberOfPredictors)), logLikelihood(
        new PRECISION(0)), mklWrapper(), betaCoefficentsOldHost(new Container::PinnedHostVector(numberOfPredictors)), predictorsDevice(
        lrConfiguration.getPredictors()), outcomesDevice(lrConfiguration.getOutcomes()), probabilitesDevice(
        lrConfiguration.getProbabilites()), scoresDevice(lrConfiguration.getScores()), workMatrixNxMDevice(
        lrConfiguration.getWorkMatrixNxM()), workVectorNx1Device(lrConfiguration.getWorkVectorNx1()), sigma(
        new PinnedHostVector(numberOfPredictors)), uSVD(new PinnedHostMatrix(numberOfPredictors, numberOfPredictors)), vtSVD(
        new PinnedHostMatrix(numberOfPredictors, numberOfPredictors)), workMatrixMxMHost(
        new PinnedHostMatrix(numberOfPredictors, numberOfPredictors)), oneVector(predictorsDevice(0)) {

}

LogisticRegression::~LogisticRegression() {
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

  std::cerr << "LR start" << std::endl;

  for(iterationNumber = 0; iterationNumber < maxIterations; ++iterationNumber){
    std::cerr << "iter " << iterationNumber << std::endl;

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
    }
  } /* for iterationNumber */
  std::cerr << "LR end" << std::endl;

  delete diffSumHost;

  kernelWrapper.syncStream();
  return new LogisticRegressionResult(betaCoefficentsHost, informationMatrixHost, inverseInformationMatrixHost,
      iterationNumber, *logLikelihood);
}

void LogisticRegression::calcuateProbabilites(const DeviceMatrix& predictorsDevice, DeviceVector& betaCoefficentsDevice,
    DeviceVector& probabilitesDevice, DeviceVector& workVectorNx1Device) {
  std::cerr << "calcuateProbabilites start" << std::endl;

  kernelWrapper.matrixVectorMultiply(predictorsDevice, betaCoefficentsDevice, workVectorNx1Device);
  kernelWrapper.logisticTransform(workVectorNx1Device, probabilitesDevice);

  std::cerr << "calcuateProbabilites end" << std::endl;
}

void LogisticRegression::calculateScores(const DeviceMatrix& predictorsDevice, const DeviceVector& outcomesDevice,
    DeviceVector& probabilitesDevice, DeviceVector& scoresDevice, DeviceVector& workVectorNx1Device) {
  std::cerr << "calculateScores start" << std::endl;

  kernelWrapper.elementWiseDifference(outcomesDevice, probabilitesDevice, workVectorNx1Device);
  kernelWrapper.matrixTransVectorMultiply(predictorsDevice, workVectorNx1Device, scoresDevice);

  std::cerr << "calculateScores end" << std::endl;
}

void LogisticRegression::calculateInformationMatrix(const DeviceMatrix& predictorsDevice,
    DeviceVector& probabilitesDevice, DeviceVector& workVectorNx1Device, DeviceMatrix& informationMatrixDevice,
    DeviceMatrix& workMatrixNxMDevice) {
  std::cerr << "calculateInformationMatrix start" << std::endl;

  kernelWrapper.probabilitesMultiplyProbabilites(probabilitesDevice, workVectorNx1Device);
  kernelWrapper.columnByColumnMatrixVectorElementWiseMultiply(predictorsDevice, workVectorNx1Device,
      workMatrixNxMDevice);
  kernelWrapper.matrixTransMatrixMultiply(predictorsDevice, workMatrixNxMDevice, informationMatrixDevice);

  std::cerr << "calculateInformationMatrix end" << std::endl;
}

void LogisticRegression::invertInformationMatrix(HostMatrix& informationMatrixHost,
    HostMatrix& inverseInformationMatrixHost, HostMatrix& uSVD, HostVector& sigma, HostMatrix& vtSVD,
    HostMatrix& workMatrixMxMHost) {

  std::cerr << "invertInformationMatrix start" << std::endl;
  int size = informationMatrixHost.getNumberOfRows();

  mklWrapper.svd(informationMatrixHost, uSVD, sigma, vtSVD);

  for(int i = 0; i < size; ++i){
    PRECISION inverseSigma;
    if(abs(sigma(i)) > 1e-7){
      inverseSigma = 1 / sigma(i);
    }else{
      inverseSigma = 0;
    }
    std::cerr << inverseSigma << std::endl;

    //col i ifrån uSVD*inverseSigma läggs i rad i i work
    //cblas_ FIXME
    for(int k = 0; k < size; ++k){
      workMatrixMxMHost(i, k) = inverseSigma * uSVD(k, i); //k i or i k?
    }
  }

  mklWrapper.matrixTransMatrixMultiply(vtSVD, workMatrixMxMHost, inverseInformationMatrixHost, 1, 0);

  std::cerr << "invertInformationMatrix end" << std::endl;
}

void LogisticRegression::calculateNewBeta(HostMatrix& inverseInformationMatrixHost, HostVector& scoresHost,
    HostVector& betaCoefficentsHost) {
  std::cerr << "calculateNewBeta start" << std::endl;

  //beta=inv*scores+beta
  mklWrapper.matrixVectorMultiply(inverseInformationMatrixHost, scoresHost, betaCoefficentsHost, 1, 1);

  std::cerr << "calculateNewBeta end" << std::endl;
}

void LogisticRegression::calculateDifference(HostVector& betaCoefficentsHost, HostVector& betaCoefficentsOldHost,
    PRECISION* diffSumHost) {
  std::cerr << "calculateDifference start" << std::endl;

  mklWrapper.differenceElememtWise(betaCoefficentsHost, betaCoefficentsOldHost);
  mklWrapper.absoluteSum(betaCoefficentsOldHost, diffSumHost);

  std::cerr << "calculateDifference end" << std::endl;
}

void LogisticRegression::calculateLogLikelihood(const DeviceVector& outcomesDevice, const DeviceVector& oneVector,
    DeviceVector& probabilitesDevice, DeviceVector& workVectorNx1Device, PRECISION* logLikelihood) {
  std::cerr << "likelihood start" << std::endl;

  kernelWrapper.logLikelihoodParts(outcomesDevice, probabilitesDevice, workVectorNx1Device);
  kernelWrapper.sumResultToHost(workVectorNx1Device, oneVector, logLikelihood);

  std::cerr << "likelihood end" << std::endl;
}

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
