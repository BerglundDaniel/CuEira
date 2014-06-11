#ifndef LOGISTICREGRESSIONCONFIGURATION_H_
#define LOGISTICREGRESSIONCONFIGURATION_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <Configuration.h>
#include <CudaAdapter.cu>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <CudaException.h>
#include <CublasException.h>
#include <DimensionMismatch.h>
#include <HostToDevice.h>
#include <PinnedHostVector.h>
#include <KernelWrapper.h>

namespace CuEira {
namespace Model {
namespace LogisticRegression {

using namespace CuEira::Container;
using namespace CuEira::CUDA;

/**
 * This is class is responsible for making the needed transfers of the data to the GPU and putting everything at the correct place
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LogisticRegressionConfiguration {
public:
  /**
   * This
   */
  LogisticRegressionConfiguration(const Configuration& configuration, const HostToDevice& hostToDevice,
      const DeviceVector& deviceOutcomes, const KernelWrapper& kernelWrapper);

  /**
   * This
   */
  LogisticRegressionConfiguration(const Configuration& configuration, const HostToDevice& hostToDevice,
      const DeviceVector& deviceOutcomes, const KernelWrapper& kernelWrapper, const HostMatrix& covariates);

  /**
   * This
   */
  virtual ~LogisticRegressionConfiguration();

  void setEnvironmentFactor(const HostVector& environmentData);
  void setSNP(const HostVector& snpData);
  void setInteraction(const HostVector& interactionVector);
  void setBetaCoefficents(const HostVector& betaCoefficents);

  int getNumberOfRows() const;
  int getNumberOfPredictors() const;
  int getNumberOfMaxIterations() const;
  double getConvergenceThreshold() const;
  const KernelWrapper& getKernelWrapper() const;

  const DeviceMatrix& getPredictors()const;
  const DeviceVector& getOutcomes()const;
  DeviceVector& getProbabilites();
  DeviceVector& getScores();
  DeviceMatrix& getInformationMatrix();
  DeviceVector& getBetaCoefficents();

  DeviceMatrix& getWorkMatrixNxM();
  DeviceVector& getWorkVectorNx1();

private:
  void transferIntercept();

  const Configuration& configuration;
  const HostToDevice& hostToDevice;
  const int numberOfRows;
  const int numberOfPredictors;
  const KernelWrapper& kernelWrapper;
  DeviceMatrix* devicePredictors;
  const DeviceVector& deviceOutcomes;
  int maxIterations;
  double convergenceThreshold;
  bool usingCovariates;
  PRECISION* devicePredictorsMemoryPointer;

  DeviceVector* betaCoefficentsDevice;
  DeviceVector* probabilitesDevice;
  DeviceVector* scoresDevice;
  DeviceMatrix* informationMatrixDevice;
  DeviceMatrix* workMatrixNxMDevice;
  DeviceVector* workVectorNx1Device;
};

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif /* LOGISTICREGRESSIONCONFIGURATION_H_ */