#ifndef LOGISTICREGRESSION_H_
#define LOGISTICREGRESSION_H_

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <CudaAdapter.cu>
#include <KernelWrapper.h>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <CudaException.h>
#include <CublasException.h>
#include <DimensionMismatch.h>
#include <LogisticRegressionConfiguration.h>
#include <HostToDevice.h>
#include <DeviceToHost.h>
#include <MKLWrapper.h>
#include <LogisticRegressionResult.h>

namespace CuEira {
namespace Model {
namespace LogisticRegression {
class LogisticRegressionTest;

using namespace CuEira::Container;
using namespace CuEira::CUDA;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LogisticRegression {
  friend LogisticRegressionTest;
  FRIEND_TEST(LogisticRegressionTest, calcuateProbabilites);
  FRIEND_TEST(LogisticRegressionTest, calculateScores);
  FRIEND_TEST(LogisticRegressionTest, calculateInformationMatrix);
  FRIEND_TEST(LogisticRegressionTest, invertInformationMatrix);
  FRIEND_TEST(LogisticRegressionTest, calculateNewBeta);
  FRIEND_TEST(LogisticRegressionTest, calculateDifference);
  FRIEND_TEST(LogisticRegressionTest, calculateLogLikelihood);
public:
  LogisticRegression(LogisticRegressionConfiguration& lrConfiguration, const HostToDevice& hostToDevice,
      const DeviceToHost& deviceToHost);
  virtual ~LogisticRegression();

  /**
   * Calculate the model
   */
  LogisticRegressionResult* calculate();

private:
  void calcuateProbabilites(const DeviceMatrix& predictorsDevice, const DeviceVector& betaCoefficentsDevice,
      DeviceVector& probabilitesDevice, DeviceVector& workVectorNx1Device);
  void calculateScores(const DeviceMatrix& predictorsDevice, const DeviceVector& outcomesDevice,
      const DeviceVector& probabilitesDevice, DeviceVector& scoresDevice, DeviceVector& workVectorNx1Device);
  void calculateInformationMatrix(const DeviceMatrix& predictorsDevice, const DeviceVector& probabilitesDevice,
      DeviceVector& workVectorNx1Device, DeviceMatrix& informationMatrixDevice, DeviceMatrix& workMatrixNxMDevice);
  void invertInformationMatrix(HostMatrix& informationMatrixHost, HostMatrix& inverseInformationMatrixHost,
     HostMatrix& uSVD, HostVector& sigma, HostMatrix& vtSVD, HostMatrix& workMatrixMxMHost);
  void calculateNewBeta(HostMatrix& inverseInformationMatrixHost, HostVector& scoresHost,
      HostVector& betaCoefficentsHost);
  void calculateDifference(const HostVector& betaCoefficentsHost, HostVector& betaCoefficentsOldHost, PRECISION* diffSumHost);
  void calculateLogLikelihood(const DeviceVector& outcomesDevice, const DeviceVector& oneVector,
      const DeviceVector& probabilitesDevice, DeviceVector& workVectorNx1Device, PRECISION* logLikelihood);

  MKLWrapper mklWrapper;
  const HostToDevice& hostToDevice;
  const DeviceToHost& deviceToHost;
  LogisticRegressionConfiguration& lrConfiguration;
  const int numberOfRows;
  const int numberOfPredictors;
  const int maxIterations;
  const double convergenceThreshold;
  const KernelWrapper& kernelWrapper;
  int iterationNumber;
  PRECISION* logLikelihood;

  const Container::DeviceMatrix& predictorsDevice;
  const Container::DeviceVector& outcomesDevice;
  const Container::DeviceVector* oneVector; //Vector of length numberOfIndividuals with just ones. To save space its the same as the first column in predictors
  Container::DeviceMatrix& informationMatrixDevice;
  Container::DeviceVector& betaCoefficentsDevice;
  Container::DeviceVector& probabilitesDevice;
  Container::DeviceVector& scoresDevice;
  Container::DeviceMatrix& workMatrixNxMDevice;
  Container::DeviceVector& workVectorNx1Device;

  Container::HostVector* scoresHost;
  Container::HostVector* betaCoefficentsOldHost;
  Container::HostVector* sigma;
  Container::HostMatrix* uSVD;
  Container::HostMatrix* vtSVD;
  Container::HostMatrix* workMatrixMxMHost;
};

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif /* LOGISTICREGRESSION_H_ */
