#include "LogisticRegression.h"

namespace CuEira {
namespace CUDA {
namespace LogisticRegression {

LogisticRegression::LogisticRegression(const LogisticRegressionConfiguration& lrConfiguration) :
    kernelWrapper(lrConfiguration.getKernelWrapper()), lrConfiguration(lrConfiguration), maxIterations(
        lrConfiguration.getNumberOfMaxIterations()), convergenceThreshold(lrConfiguration.getConvergenceThreshold()), numberOfRows(
        lrConfiguration.getNumberOfRows()), numberOfPredictors(lrConfiguration.getNumberOfPredictors()), informationMatrixDevice(
        lrConfiguration.getInformationMatrix()), betaCoefficentsDevice(lrConfiguration.getBetaCoefficents()) {

  PRECISION* diffSumHost = new PRECISION(0);

  const Container::DeviceMatrix& predictorsDevice = lrConfiguration.getPredictors();
  const Container::DeviceVector& outcomesDevice = lrConfiguration.getOutcomes();
  const Container::DeviceVector& probabilitesDevice = lrConfiguration.getProbabilites();
  const Container::DeviceVector& scoresDevice = lrConfiguration.getScores();

  const Container::DeviceMatrix& workMatrixNxMDevice = lrConfiguration.getWorkMatrixNxM();
  const Container::DeviceVector& workVectorNx1Device = lrConfiguration.getWorkVectorNx1();
  const Container::DeviceVector& workVectorMx1Device = lrConfiguration.getWorkVectorMx1();
  const Container::DeviceVector& betaCoefficentsOldDevice = lrConfiguration.getBetaCoefficentsOld();

  const Container::DeviceMatrix& uSVD = lrConfiguration.getUSVD();
  const Container::DeviceMatrix& vtSVD = lrConfiguration.getVtSVD();
  const Container::DeviceVector& sigmaSVD = lrConfiguration.getSigmaSVD();

  //u v mxm
  //sigma mx1

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
    kernelWrapper.columnByColumnMatrixVectorMultiply(predictorsDevice, workVectorNx1Device, workMatrixNxMDevice);
    kernelWrapper.matrixTransMatrixMultiply(predictorsDevice, workMatrixNxMDevice, informationMatrixDevice);

    //Inverse information matrix and calculate new beta
    //NOTE Using betaCoefficentsDevice as an work area here since it's values are in betaCoefficentsOldDevice anyway
    kernelWrapper.svd(informationMatrixDevice, uSVD, sigmaSVD, vtSVD);
    kernelWrapper.matrixTransVectorMultiply(uSVD, scoresDevice, workVectorMx1Device);
    kernelWrapper.elementWiseDivision(workVectorMx1Device, sigmaSVD, betaCoefficentsDevice); //Overwriting beta
    kernelWrapper.matrixTransVectorMultiply(vtSVD, betaCoefficentsDevice, workVectorMx1Device);
    kernelWrapper.elementWiseAddition(betaCoefficentsOldDevice, workVectorMx1Device, betaCoefficentsDevice);

    //Calculate difference
    kernelWrapper.absoluteDifference(betaCoefficentsDevice, betaCoefficentsOldDevice, workVectorNx1Device);
    kernelWrapper.sumResultToHost(workVectorNx1Device, diffSumHost);

    kernelWrapper.syncStream();
    if(*diffSum < convergenceThreshold){

      //Calculate loglikelihood
      kernelWrapper.logLikelihoodParts(outcomesDevice, probabilitesDevice, workVectorNx1Device);
      kernelWrapper.sumResultToHost(workVectorNx1Device, logLikelihood);

      break;
    }
  } /* for iterationNumber */

  kernelWrapper.syncStream();
}

LogisticRegression::~LogisticRegression() {

}

const DeviceVector& LogisticRegression::getBeta() const {
  return betaCoefficentsDevice;
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
