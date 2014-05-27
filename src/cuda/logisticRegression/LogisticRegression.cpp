#include "LogisticRegression.h"

namespace CuEira {
namespace CUDA {
namespace LogisticRegression {

LogisticRegression::LogisticRegression(LogisticRegressionConfiguration& lrConfiguration) :
    kernelWrapper(lrConfiguration.getKernelWrapper()), lrConfiguration(lrConfiguration), maxIterations(
        lrConfiguration.getNumberOfMaxIterations()), convergenceThreshold(lrConfiguration.getConvergenceThreshold()), numberOfRows(
        lrConfiguration.getNumberOfRows()), numberOfPredictors(lrConfiguration.getNumberOfPredictors()), informationMatrixDevice(
        lrConfiguration.getInformationMatrix()), betaCoefficentsDevice(lrConfiguration.getBetaCoefficents()), inverseInformationMatrixDevice(
        lrConfiguration.getInverseMatrix()) {

  PRECISION* diffSumHost = new PRECISION(0);

  const Container::DeviceMatrix& predictorsDevice = lrConfiguration.getPredictors();
  const Container::DeviceVector& outcomesDevice = lrConfiguration.getOutcomes();
  Container::DeviceVector& probabilitesDevice = lrConfiguration.getProbabilites();
  Container::DeviceVector& scoresDevice = lrConfiguration.getScores();

  Container::DeviceMatrix& workMatrixNxMDevice = lrConfiguration.getWorkMatrixNxM();
  Container::DeviceVector& workVectorNx1Device = lrConfiguration.getWorkVectorNx1();
  Container::DeviceVector& workVectorMx1Device = lrConfiguration.getWorkVectorMx1();
  Container::DeviceVector& betaCoefficentsOldDevice = lrConfiguration.getBetaCoefficentsOld();
  Container::DeviceMatrix& workMatrixMxMDevice = lrConfiguration.getWorkMatrixMxM();

  Container::DeviceMatrix& uSVD = lrConfiguration.getUSVD();
  Container::DeviceMatrix& vtSVD = lrConfiguration.getVtSVD();
  Container::DeviceVector& sigmaSVD = lrConfiguration.getSigmaSVD();

  for(iterationNumber = 0; iterationNumber < maxIterations; ++iterationNumber){
    //Copy beta to old beta
    kernelWrapper.copyVector(betaCoefficentsDevice, betaCoefficentsOldDevice);

    //Calculate probabilities
    kernelWrapper.matrixVectorMultiply(predictorsDevice, betaCoefficentsDevice, workVectorNx1Device);
    kernelWrapper.logisticTransform(workVectorNx1Device, probabilitesDevice);

    //Calculate scores
    kernelWrapper.elementWiseDifference(outcomesDevice, probabilitesDevice, workVectorNx1Device);
    kernelWrapper.matrixTransVectorMultiply(predictorsDevice, workVectorNx1Device, scoresDevice);

    //Calculate information matrix
    kernelWrapper.probabilitesMultiplyProbabilites(probabilitesDevice, workVectorNx1Device);
    kernelWrapper.columnByColumnMatrixVectorElementWiseMultiply(predictorsDevice, workVectorNx1Device, workMatrixNxMDevice);
    kernelWrapper.matrixTransMatrixMultiply(predictorsDevice, workMatrixNxMDevice, informationMatrixDevice);

    //Inverse information matrix

    //Do the rest of this iterations stuff on cpu? FIXME
    kernelWrapper.svd(informationMatrixDevice, uSVD, sigmaSVD, vtSVD);

    kernelWrapper.matrixTransRowByRowInverseSigma(vtSVD, sigmaSVD, workMatrixMxMDevice);
    kernelWrapper.matrixTransMatrixMultiply(uSVD, workMatrixMxMDevice, inverseInformationMatrixDevice);

    //Calculate new beta
    kernelWrapper.matrixVectorMultiply(inverseInformationMatrixDevice, scoresDevice, workVectorMx1Device);
    kernelWrapper.elementWiseAddition(betaCoefficentsOldDevice, workVectorMx1Device, betaCoefficentsDevice);

    //Calculate difference
    kernelWrapper.elementWiseAbsoluteDifference(betaCoefficentsDevice, betaCoefficentsOldDevice, workVectorNx1Device);
    kernelWrapper.sumResultToHost(workVectorNx1Device, diffSumHost);

    kernelWrapper.syncStream();
    if(*diffSumHost < convergenceThreshold){

      //Calculate loglikelihood
      kernelWrapper.logLikelihoodParts(outcomesDevice, probabilitesDevice, workVectorNx1Device);
      kernelWrapper.sumResultToHost(workVectorNx1Device, logLikelihood);

      break;
    }
  } /* for iterationNumber */

  delete diffSumHost;
  kernelWrapper.syncStream();
}

LogisticRegression::~LogisticRegression() {

}

const DeviceVector& LogisticRegression::getBeta() const {
  return betaCoefficentsDevice;
}

const DeviceVector& LogisticRegression::getCovarianceMatrix() const {
  return inverseInformationMatrixDevice;
}

const DeviceMatrix& LogisticRegression::getInformationMatrix() const {
  return informationMatrixDevice;
}

int LogisticRegression::getNumberOfIterations() const {
  return iterationNumber;
}

PRECISION LogisticRegression::getLogLikelihood() const {
  return *logLikelihood;
}

} /* namespace LogisticRegression */
} /* namespace CUDA */
} /* namespace CuEira */
