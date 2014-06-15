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
        new PinnedHostMatrix(numberOfPredictors, numberOfPredictors)), scoresHost(new PinnedHostVector(numberOfRows)), inverseInformationMatrixHost(
        new Container::PinnedHostMatrix(numberOfPredictors, numberOfPredictors)) {

  double* diffSumHost = new double(0);
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

  for(iterationNumber = 0; iterationNumber < maxIterations; ++iterationNumber){
    //Calculate probabilities
    kernelWrapper.matrixVectorMultiply(predictorsDevice, betaCoefficentsDevice, workVectorNx1Device);
    kernelWrapper.logisticTransform(workVectorNx1Device, probabilitesDevice);

    //Calculate scores
    kernelWrapper.elementWiseDifference(outcomesDevice, probabilitesDevice, workVectorNx1Device);
    kernelWrapper.matrixTransVectorMultiply(predictorsDevice, workVectorNx1Device, scoresDevice);

    //Calculate information matrix
    kernelWrapper.probabilitesMultiplyProbabilites(probabilitesDevice, workVectorNx1Device);
    kernelWrapper.columnByColumnMatrixVectorElementWiseMultiply(predictorsDevice, workVectorNx1Device,
        workMatrixNxMDevice);
    kernelWrapper.matrixTransMatrixMultiply(predictorsDevice, workMatrixNxMDevice, informationMatrixDevice);

    //Copy beta to old beta
#ifdef DOUBLEPRECISION
    cblas_dcopy((MKL_INT)numberOfPredictors, betaCoefficentsHost->getMemoryPointer(), (MKL_INT)1,
        betaCoefficentsOldHost->getMemoryPointer(), (MKL_INT)1);
#else
    cblas_scopy((MKL_INT) numberOfPredictors, betaCoefficentsHost->getMemoryPointer(), (MKL_INT) 1,
        betaCoefficentsOldHost->getMemoryPointer(), (MKL_INT) 1);
#endif

    //Inverse information matrix
    //NOTE This part is done on CPU

    //Transfer needed data to host
    deviceToHost.transferMatrix(&informationMatrixDevice, informationMatrixHost->getMemoryPointer());
    deviceToHost.transferVector(&scoresDevice, scoresHost->getMemoryPointer());
    kernelWrapper.syncStream();

    //Invert
    MKL_INT status = LAPACKE_sgesdd(LAPACK_COL_MAJOR, 'A', numberOfPredictors, numberOfPredictors,
        informationMatrixHost->getMemoryPointer(), numberOfPredictors, sigma.getMemoryPointer(),
        uSVD.getMemoryPointer(), numberOfPredictors, vtSVD.getMemoryPointer(), numberOfPredictors);

    if(status < 0){
      throw new InvalidState("Illegal values in informatio matrix.");
    }else if(status > 0){
      std::cerr << "Warning matrix svd didn't converge." << std::endl;
    }

    for(int i = 0; i < numberOfPredictors; ++i){
      PRECISION inverseSigma;
      if(sigma(i) < 1e-10){
        inverseSigma = 1 / sigma(i);
      }else{
        inverseSigma = 0;
      }

      //col i ifrån uSVD*inverseSigma läggs i rad i i work
      //cblas_ FIXME
      for(int k = 0; k < numberOfPredictors; ++k){
        workMatrixMxMHost(i, k) = inverseSigma * uSVD(i, k);
      }
    }

    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, numberOfPredictors, numberOfPredictors, numberOfPredictors, 1,
        vtSVD.getMemoryPointer(), numberOfPredictors, workMatrixMxMHost.getMemoryPointer(), numberOfPredictors, 0,
        inverseInformationMatrixHost->getMemoryPointer(), numberOfPredictors);

    //Calculate new beta

    //beta_old=inv*scores+beta_old
    cblas_sgemv(CblasColMajor, CblasNoTrans, numberOfPredictors, numberOfPredictors, 1,
        inverseInformationMatrixHost->getMemoryPointer(), numberOfPredictors, scoresHost->getMemoryPointer(), 1, 1,
        betaCoefficentsOldHost->getMemoryPointer(), 1);

    //Calculate difference
    cblas_saxpy(numberOfPredictors, -1.0, betaCoefficentsHost->getMemoryPointer(), 1,
        betaCoefficentsOldHost->getMemoryPointer(), 1);
    *diffSumHost = cblas_sasum(numberOfPredictors, betaCoefficentsOldHost->getMemoryPointer(), 1);

    if(*diffSumHost < convergenceThreshold){

      //Calculate loglikelihood
      kernelWrapper.logLikelihoodParts(outcomesDevice, probabilitesDevice, workVectorNx1Device);
      kernelWrapper.sumResultToHost(workVectorNx1Device, logLikelihood);

      //Transfer the information matrix again since it was destroyed during the SVD.
      deviceToHost.transferMatrix(&informationMatrixDevice, informationMatrixHost->getMemoryPointer());

      break;
    }else{
      hostToDevice.transferVector(betaCoefficentsHost, betaCoefficentsDevice.getMemoryPointer());
    }
  } /* for iterationNumber */

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
