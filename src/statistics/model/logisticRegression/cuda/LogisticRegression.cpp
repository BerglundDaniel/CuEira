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
        lrConfiguration.getBetaCoefficents()), informationMatrixHost(
        new PinnedHostMatrix(numberOfPredictors, numberOfPredictors)), scoresHost(new PinnedHostVector(numberOfPredictors)), inverseInformationMatrixHost(
        new Container::PinnedHostMatrix(numberOfPredictors, numberOfPredictors)), logLikelihood(new PRECISION(0)), betaCoefficentsHost(
        deviceToHost.transferVector(&betaCoefficentsDevice)) {

  PRECISION* diffSumHost = new PRECISION(0);
  PinnedHostVector sigma(numberOfPredictors);
  PinnedHostMatrix uSVD(numberOfPredictors, numberOfPredictors);
  PinnedHostMatrix vtSVD(numberOfPredictors, numberOfPredictors);
  PinnedHostMatrix workMatrixMxMHost(numberOfPredictors, numberOfPredictors);
  Container::HostVector* betaCoefficentsOldHost = new Container::PinnedHostVector(numberOfPredictors);

  const Container::DeviceMatrix& predictorsDevice = lrConfiguration.getPredictors();
  const Container::DeviceVector& outcomesDevice = lrConfiguration.getOutcomes();
  Container::DeviceVector& probabilitesDevice = lrConfiguration.getProbabilites();
  Container::DeviceVector& scoresDevice = lrConfiguration.getScores();

  Container::DeviceMatrix& workMatrixNxMDevice = lrConfiguration.getWorkMatrixNxM();
  Container::DeviceVector& workVectorNx1Device = lrConfiguration.getWorkVectorNx1();

  MKLWrapper mklWrapper;

  std::cerr << "start" << std::endl;

  for(iterationNumber = 0; iterationNumber < maxIterations; ++iterationNumber){
    //Calculate probabilities
    std::cerr << "iter" << std::endl;
    kernelWrapper.matrixVectorMultiply(predictorsDevice, betaCoefficentsDevice, workVectorNx1Device);
    std::cerr << "s1" << std::endl;
    kernelWrapper.logisticTransform(workVectorNx1Device, probabilitesDevice);
    std::cerr << "s2" << std::endl;
    //Calculate scores
    kernelWrapper.elementWiseDifference(outcomesDevice, probabilitesDevice, workVectorNx1Device);
    std::cerr << "s3" << std::endl;
    kernelWrapper.matrixTransVectorMultiply(predictorsDevice, workVectorNx1Device, scoresDevice); //Hit
    std::cerr << "s4" << std::endl;

    //Calculate information matrix
    kernelWrapper.probabilitesMultiplyProbabilites(probabilitesDevice, workVectorNx1Device);
    std::cerr << "s5" << std::endl;
    kernelWrapper.columnByColumnMatrixVectorElementWiseMultiply(predictorsDevice, workVectorNx1Device,
        workMatrixNxMDevice);
    std::cerr << "s6" << std::endl;
    kernelWrapper.matrixTransMatrixMultiply(predictorsDevice, workMatrixNxMDevice, informationMatrixDevice);
    std::cerr << "s7" << std::endl;
    //Copy beta to old beta
    mklWrapper.copyVector(*betaCoefficentsHost, *betaCoefficentsOldHost);
    std::cerr << "s8" << std::endl;

    //Inverse information matrix
    //NOTE This part is done on CPU

    //Transfer needed data to host
    deviceToHost.transferMatrix(&informationMatrixDevice, informationMatrixHost->getMemoryPointer());
    deviceToHost.transferVector(&scoresDevice, scoresHost->getMemoryPointer());
    kernelWrapper.syncStream();
    std::cerr << "s9" << std::endl;
    std::cerr << (*scoresHost)(0) << std::endl;
    std::cerr << (*informationMatrixHost)(0, 0) << std::endl;

    //Invert
    mklWrapper.svd(*informationMatrixHost, uSVD, sigma, vtSVD);

    //FIXME
    for(int i = 0; i < numberOfPredictors; ++i){
      PRECISION inverseSigma;
      if(sigma(i) < 1e-10){
        inverseSigma = 1 / sigma(i);
      }else{
        inverseSigma = 0;
      }
      std::cerr << inverseSigma << std::endl;

      //col i ifrån uSVD*inverseSigma läggs i rad i i work
      //cblas_ FIXME
      for(int k = 0; k < numberOfPredictors; ++k){
        workMatrixMxMHost(i, k) = inverseSigma * uSVD(i, k);
      }
    }
    std::cerr << "s11" << std::endl;

    mklWrapper.matrixTransMatrixMultiply(vtSVD, workMatrixMxMHost, *inverseInformationMatrixHost, 1, 0);

    std::cerr << "s12" << std::endl;
    //Calculate new beta

    //beta=inv*scores+beta
    mklWrapper.matrixVectorMultiply(*inverseInformationMatrixHost, *scoresHost, *betaCoefficentsHost, 1, 1);

    std::cerr << "s13" << std::endl;

    //Calculate difference
    mklWrapper.differenceElememtWise(*betaCoefficentsHost, *betaCoefficentsOldHost);
    mklWrapper.absoluteSum(*betaCoefficentsOldHost, diffSumHost);

    std::cerr << "s14" << std::endl;
    std::cerr << (*betaCoefficentsHost)(0) << std::endl;

    if(*diffSumHost < convergenceThreshold){
      std::cerr << "s15" << std::endl;
      //Calculate loglikelihood
      kernelWrapper.logLikelihoodParts(outcomesDevice, probabilitesDevice, workVectorNx1Device);
      std::cerr << "s16" << std::endl;
      kernelWrapper.sumResultToHost(workVectorNx1Device, logLikelihood);
      std::cerr << "s17" << std::endl;
      //Transfer the information matrix again since it was destroyed during the SVD.
      deviceToHost.transferMatrix(&informationMatrixDevice, informationMatrixHost->getMemoryPointer());

      break;
    }else{
      hostToDevice.transferVector(betaCoefficentsHost, betaCoefficentsDevice.getMemoryPointer());
    }
  } /* for iterationNumber */
  std::cerr << "end" << std::endl;
  delete diffSumHost;
  delete betaCoefficentsOldHost;

  kernelWrapper.syncStream();
}

LogisticRegression::~LogisticRegression() {
  delete betaCoefficentsHost;
  delete inverseInformationMatrixHost;
  delete informationMatrixHost;
  delete scoresHost;
}

HostVector* LogisticRegression::stealBeta() {
  HostVector* tmp = betaCoefficentsHost;
  betaCoefficentsHost = nullptr;
  return tmp;
}

const HostMatrix& LogisticRegression::getCovarianceMatrix() const {
  return *inverseInformationMatrixHost;
}

const HostMatrix& LogisticRegression::getInformationMatrix() const {
  return *informationMatrixHost;
}

int LogisticRegression::getNumberOfIterations() const {
  return iterationNumber;
}

PRECISION LogisticRegression::getLogLikelihood() const {
  return *logLikelihood;
}

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
