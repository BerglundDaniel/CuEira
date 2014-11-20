#ifndef CUDALOGISTICREGRESSION_H_
#define CUDALOGISTICREGRESSION_H_

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <LogisticRegression.h>
#include <CudaAdapter.cu>
#include <KernelWrapper.h>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <CudaException.h>
#include <CublasException.h>
#include <DimensionMismatch.h>
#include <CudaLogisticRegressionConfiguration.h>
#include <HostToDevice.h>
#include <DeviceToHost.h>
#include <MKLWrapper.h>
#include <LogisticRegressionResult.h>
#include <ModelResult.h>
#include <PinnedHostVector.h>
#include <PinnedHostMatrix.h>
#include <RegularHostVector.h>
#include <RegularHostMatrix.h>

#ifdef PROFILE
#include <boost/chrono/chrono_io.hpp>
#include <thread>
#include <mutex>
#endif

namespace CuEira {
namespace Model {
namespace LogisticRegression {
namespace CUDA {
class CudaLogisticRegressionTest;

using namespace CuEira::CUDA;
using namespace CuEira::Container;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaLogisticRegression: public LogisticRegression {
  friend CudaLogisticRegressionTest;
  FRIEND_TEST(CudaLogisticRegressionTest, calcuateProbabilites);
  FRIEND_TEST(CudaLogisticRegressionTest, calculateScores);
  FRIEND_TEST(CudaLogisticRegressionTest, calculateInformationMatrix);
  FRIEND_TEST(CudaLogisticRegressionTest, calculateLogLikelihood);
public:
  CudaLogisticRegression(CudaLogisticRegressionConfiguration* lrConfiguration);
  virtual ~CudaLogisticRegression();

  /**
   * Calculate the model
   */
  virtual LogisticRegressionResult* calculate();

  CudaLogisticRegression(const CudaLogisticRegression&) = delete;
  CudaLogisticRegression(CudaLogisticRegression&&) = delete;
  CudaLogisticRegression& operator=(const CudaLogisticRegression&) = delete;
  CudaLogisticRegression& operator=(CudaLogisticRegression&&) = delete;

protected:
  CudaLogisticRegression(); //For the mock

private:
  void calcuateProbabilites(const DeviceMatrix& predictorsDevice, const DeviceVector& betaCoefficentsDevice,
      DeviceVector& probabilitesDevice, DeviceVector& workVectorNx1Device);
  void calculateScores(const DeviceMatrix& predictorsDevice, const DeviceVector& outcomesDevice,
      const DeviceVector& probabilitesDevice, DeviceVector& scoresDevice, DeviceVector& workVectorNx1Device);
  void calculateInformationMatrix(const DeviceMatrix& predictorsDevice, const DeviceVector& probabilitesDevice,
      DeviceVector& workVectorNx1Device, DeviceMatrix& informationMatrixDevice, DeviceMatrix& workMatrixNxMDevice);
  void calculateLogLikelihood(const DeviceVector& outcomesDevice, const DeviceVector& oneVector,
      const DeviceVector& probabilitesDevice, DeviceVector& workVectorNx1Device, PRECISION& logLikelihood);

  CudaLogisticRegressionConfiguration* lrConfiguration;
  const HostToDevice* hostToDevice;
  const DeviceToHost* deviceToHost;
  const KernelWrapper* kernelWrapper;

  const Container::DeviceMatrix* predictorsDevice;
  const Container::DeviceVector* outcomesDevice;
  const Container::DeviceVector* oneVector; //Vector of length numberOfIndividuals with just ones. To save space its the same as the first column in predictors

  //NOTE The config class owns all device memory while this class owns the host classes
  Container::DeviceMatrix* informationMatrixDevice;
  Container::DeviceVector* betaCoefficentsDevice;
  Container::DeviceVector* probabilitesDevice;
  Container::DeviceVector* scoresDevice;
  Container::DeviceMatrix* workMatrixNxMDevice;
  Container::DeviceVector* workVectorNx1Device;

  const Container::PinnedHostVector* defaultBetaCoefficents; //LRConfig owns it

#ifdef PROFILE
  static boost::chrono::duration<long long, boost::nano> timeSpentTotal;
  static boost::chrono::duration<long long, boost::nano> timeSpentGPU;
  static boost::chrono::duration<long long, boost::nano> timeSpentCPU;
  static bool firstDestroy;
  static std::mutex mutex;
#endif
};

} /* namespace CUDA */
} /* namespace CudaLogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif /* CUDALOGISTICREGRESSION_H_ */
