#include "CudaLogisticRegression.h"

namespace CuEira {
namespace Model {
namespace LogisticRegression {
namespace CUDA {

#ifdef PROFILE
boost::chrono::duration<long long, boost::nano> CudaLogisticRegression::timeSpentTotal;
boost::chrono::duration<long long, boost::nano> CudaLogisticRegression::timeSpentCPU;

float CudaLogisticRegression::timeSpentGPU;
float CudaLogisticRegression::timeSpentTransferFromDevice;
float CudaLogisticRegression::timeSpentTransferToDevice;
#endif

CudaLogisticRegression::CudaLogisticRegression(CudaLogisticRegressionConfiguration* lrConfiguration) :
    LogisticRegression(lrConfiguration), lrConfiguration(lrConfiguration), stream(lrConfiguration->getStream()), informationMatrixDevice(
        &lrConfiguration->getInformationMatrix()), betaCoefficentsDevice(&lrConfiguration->getBetaCoefficents()), predictorsDevice(
        &lrConfiguration->getPredictors()), outcomesDevice(&lrConfiguration->getOutcomes()), probabilitesDevice(
        &lrConfiguration->getProbabilites()), scoresDevice(&lrConfiguration->getScores()), workMatrixNxMDevice(
        &lrConfiguration->getWorkMatrixNxM()), workVectorNx1Device(&lrConfiguration->getWorkVectorNx1()), oneVector(
        (*predictorsDevice)(0)), defaultBetaCoefficents(&lrConfiguration->getDefaultBetaCoefficents()){

}

CudaLogisticRegression::CudaLogisticRegression(const Stream& stream) :
    LogisticRegression(), informationMatrixDevice(nullptr), betaCoefficentsDevice(nullptr), predictorsDevice(nullptr), outcomesDevice(
        nullptr), probabilitesDevice(nullptr), scoresDevice(nullptr), workMatrixNxMDevice(nullptr), workVectorNx1Device(
        nullptr), lrConfiguration(nullptr), oneVector(nullptr), defaultBetaCoefficents(nullptr), stream(stream){

}

CudaLogisticRegression::~CudaLogisticRegression(){
  delete oneVector;
}

LogisticRegressionResult* CudaLogisticRegression::calculate(){
#ifdef PROFILE
  boost::chrono::system_clock::time_point before = boost::chrono::system_clock::now();
  Event* beforeTransferElse = nullptr;
  Event* afterTransferElse = nullptr;
#endif

  PRECISION diffSumHost = 0;
  logLikelihood = 0;

  Container::PinnedHostVector* betaCoefficentsHost = new Container::PinnedHostVector(numberOfPredictors);
  Blas::copyVector(*defaultBetaCoefficents, *betaCoefficentsHost);

#ifdef PROFILE
  Event beforeInitTransfer(stream);
#endif
  transferVector(stream, *defaultBetaCoefficents, betaCoefficentsDevice->getMemoryPointer());
#ifdef PROFILE
  Event afterInitTransfer(stream);
#endif

#ifdef FERMI
  stream.syncStream();
#endif

  Container::PinnedHostMatrix* informationMatrixHost = new Container::PinnedHostMatrix(numberOfPredictors,
      numberOfPredictors);
  Container::RegularHostMatrix* inverseInformationMatrixHost = new Container::RegularHostMatrix(numberOfPredictors,
      numberOfPredictors);

  int iterationNumber = 1;
  for(iterationNumber = 1; iterationNumber < maxIterations; ++iterationNumber){
#ifdef PROFILE
    Event beforeKernel(stream);
#endif

    calcuateProbabilites(*predictorsDevice, *betaCoefficentsDevice, *probabilitesDevice, *workVectorNx1Device);

    calculateScores(*predictorsDevice, *outcomesDevice, *probabilitesDevice, *scoresDevice, *workVectorNx1Device);

    calculateInformationMatrix(*predictorsDevice, *probabilitesDevice, *workVectorNx1Device, *informationMatrixDevice,
        *workMatrixNxMDevice);

#ifdef PROFILE
    Event afterKernel(stream);
#endif

    //Transfer needed data to host
    transferMatrix(stream, *informationMatrixDevice, informationMatrixHost->getMemoryPointer());
#ifdef FERMI
    stream.syncStream();
#endif

    transferVector(stream, *scoresDevice, scoresHost->getMemoryPointer());

#ifdef PROFILE
    Event afterTransfer(stream);
#endif

    stream.syncStream();

#ifdef PROFILE
    timeSpentGPU+=afterKernel - beforeKernel;
    timeSpentTransferFromDevice+= afterTransfer - afterKernel;

    boost::chrono::system_clock::time_point beforeCPU = boost::chrono::system_clock::now();
#endif

    //Copy beta to old beta
    Blas::copyVector(*betaCoefficentsHost, *betaCoefficentsOldHost);

    invertInformationMatrix(*informationMatrixHost, *inverseInformationMatrixHost, *uSVD, *sigma, *vtSVD,
        *workMatrixMxMHost);

    calculateNewBeta(*inverseInformationMatrixHost, *scoresHost, *betaCoefficentsHost);

    calculateDifference(*betaCoefficentsHost, *betaCoefficentsOldHost, diffSumHost);

#ifdef PROFILE
    boost::chrono::system_clock::time_point afterCPU = boost::chrono::system_clock::now();
    timeSpentCPU+=afterCPU - beforeCPU;
#endif

#ifdef PROFILE
    if(afterTransferElse!=nullptr){
      timeSpentTransferFromDevice+= *afterTransferElse - *beforeTransferElse;
    }
#endif

    if(diffSumHost < convergenceThreshold){
#ifdef PROFILE
      Event beforeKernelBreak(stream);
#endif
      calculateLogLikelihood(*outcomesDevice, *oneVector, *probabilitesDevice, *workVectorNx1Device, logLikelihood);

#ifdef PROFILE
      Event afterKernelBreak(stream);
#endif

      //Transfer the information matrix again since it was destroyed during the SVD.
      transferMatrix(stream, *informationMatrixDevice, informationMatrixHost->getMemoryPointer());

#ifdef PROFILE
      Event afterTransferBreak(stream);
#endif

      stream.syncStream();

#ifdef PROFILE
      timeSpentGPU+= afterKernelBreak - beforeKernelBreak;
      timeSpentTransferFromDevice+= afterTransferBreak - afterKernelBreak;
#endif

      break;
    }else{
#ifdef PROFILE
      delete beforeTransferElse;
      delete afterTransferElse;

      beforeTransferElse = new Event(stream);
#endif
      transferVector(stream, *betaCoefficentsHost, betaCoefficentsDevice->getMemoryPointer());
#ifdef PROFILE
      afterTransferElse = new Event(stream);
#endif

#ifdef FERMI
      stream.syncStream();
#endif
    }
  } /* for iterationNumber */

#ifdef PROFILE
  timeSpentTransferToDevice+= afterInitTransfer - beforeInitTransfer;

  boost::chrono::system_clock::time_point after = boost::chrono::system_clock::now();
  timeSpentTotal+=after - before;

  if(lrConfiguration->beforeCov != nullptr){
    timeSpentTransferToDevice+= *lrConfiguration->afterCov - *lrConfiguration->beforeCov;

    delete lrConfiguration->beforeCov;
    delete lrConfiguration->afterCov;
    lrConfiguration->beforeCov = nullptr;
    lrConfiguration->afterCov = nullptr;
  }

  if(lrConfiguration->beforeIntercept != nullptr){
    timeSpentTransferToDevice+= *lrConfiguration->afterIntercept - *lrConfiguration->beforeIntercept;

    delete lrConfiguration->beforeIntercept;
    delete lrConfiguration->afterIntercept;
    lrConfiguration->beforeIntercept = nullptr;
    lrConfiguration->afterIntercept = nullptr;
  }

  if(lrConfiguration->beforeSNP != nullptr){
    timeSpentTransferToDevice+= *lrConfiguration->afterSNP - *lrConfiguration->beforeSNP;

    delete lrConfiguration->beforeSNP;
    delete lrConfiguration->afterSNP;
    lrConfiguration->beforeSNP = nullptr;
    lrConfiguration->afterSNP = nullptr;
  }

  if(lrConfiguration->beforeEnv != nullptr){
    timeSpentTransferToDevice+= *lrConfiguration->afterEnv - *lrConfiguration->beforeEnv;

    delete lrConfiguration->beforeEnv;
    delete lrConfiguration->afterEnv;
    lrConfiguration->beforeEnv = nullptr;
    lrConfiguration->afterEnv = nullptr;
  }

  if(lrConfiguration->beforeInter != nullptr){
    timeSpentTransferToDevice+= *lrConfiguration->afterInter - *lrConfiguration->beforeInter;

    delete lrConfiguration->beforeInter;
    delete lrConfiguration->afterInter;
    lrConfiguration->beforeInter = nullptr;
    lrConfiguration->afterInter = nullptr;
  }

  if(iterationNumber != 1){
    timeSpentTransferFromDevice+= *afterTransferElse - *beforeTransferElse;
  }
  delete beforeTransferElse;
  delete afterTransferElse;

#endif

  return new LogisticRegressionResult(betaCoefficentsHost, informationMatrixHost, inverseInformationMatrixHost,
      iterationNumber, logLikelihood);
}

void CudaLogisticRegression::calcuateProbabilites(const DeviceMatrix& predictorsDevice,
    const DeviceVector& betaCoefficentsDevice, DeviceVector& probabilitesDevice, DeviceVector& workVectorNx1Device){
  Kernel::matrixVectorMultiply(stream, predictorsDevice, betaCoefficentsDevice, workVectorNx1Device);
  Kernel::logisticTransform(stream, workVectorNx1Device, probabilitesDevice);
}

void CudaLogisticRegression::calculateScores(const DeviceMatrix& predictorsDevice, const DeviceVector& outcomesDevice,
    const DeviceVector& probabilitesDevice, DeviceVector& scoresDevice, DeviceVector& workVectorNx1Device){
  Kernel::elementWiseDifference(stream, outcomesDevice, probabilitesDevice, workVectorNx1Device);
  Kernel::matrixTransVectorMultiply(stream, predictorsDevice, workVectorNx1Device, scoresDevice);
}

void CudaLogisticRegression::calculateInformationMatrix(const DeviceMatrix& predictorsDevice,
    const DeviceVector& probabilitesDevice, DeviceVector& workVectorNx1Device, DeviceMatrix& informationMatrixDevice,
    DeviceMatrix& workMatrixNxMDevice){
  Kernel::probabilitesMultiplyProbabilites(stream, probabilitesDevice, workVectorNx1Device);
  columnByColumnMatrixVectorElementWiseMultiply(predictorsDevice, workVectorNx1Device,
      workMatrixNxMDevice);
  Kernel::matrixTransMatrixMultiply(stream, predictorsDevice, workMatrixNxMDevice, informationMatrixDevice);
}

void CudaLogisticRegression::calculateLogLikelihood(const DeviceVector& outcomesDevice, const DeviceVector& oneVector,
    const DeviceVector& probabilitesDevice, DeviceVector& workVectorNx1Device, PRECISION& logLikelihood){
  Kernel::logLikelihoodParts(stream, outcomesDevice, probabilitesDevice, workVectorNx1Device);
  Kernel::sumResultToHost(stream, workVectorNx1Device, oneVector, logLikelihood);
}

void CudaLogisticRegression::columnByColumnMatrixVectorElementWiseMultiply(const DeviceMatrix& matrix,
    const DeviceVector& vector, DeviceMatrix& result) const{
#ifdef DEBUG
  if((matrix.getNumberOfRows() != vector.getNumberOfRows()) || (vector.getNumberOfRows() != result.getNumberOfRows())){
    std::ostringstream os;
    os << "Number of rows doesn't match in columnByColumnMatrixVectorElementWiseMultiply function, they are " << matrix.getNumberOfRows()
    << " , " << vector.getNumberOfRows() << " and " << result.getNumberOfRows() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }

  if(matrix.getNumberOfColumns() != result.getNumberOfColumns()){
    std::ostringstream os;
    os << "Number of columns doesn't match in columnByColumnMatrixVectorElementWiseMultiply function, they are " << matrix.getNumberOfColumns() <<
    " and " << result.getNumberOfColumns() << std::endl;
    const std::string& tmp = os.str();
    throw CudaException(tmp.c_str());
  }
#endif

  const int numberOfColumns = matrix.getNumberOfColumns();
  for(int k = 0; k < numberOfColumns; ++k){
    const DeviceVector* columnVector = matrix(k);
    DeviceVector* columnResultVector = result(k);
    Kernel::elementWiseMultiplication(stream, *columnVector, vector, *columnResultVector);

    delete columnVector;
    delete columnResultVector;
  }
}

} /* namespace CUDA */
} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */
