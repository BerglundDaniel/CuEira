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
namespace CUDA {
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
      const HostVector& outcomes, const KernelWrapper& kernelWrapper);

  /**
   * This
   */
  LogisticRegressionConfiguration(const Configuration& configuration, const HostToDevice& hostToDevice,
      const HostVector& outcomes, const KernelWrapper& kernelWrapper, const HostMatrix& covariates);

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

  const DeviceMatrix& getPredictors() const;
  const DeviceVector& getOutcomes() const;
  const DeviceVector& getProbabilites() const;
  const DeviceVector& getScores() const;
  const DeviceMatrix& getInformationMatrix() const;
  const DeviceVector& getBetaCoefficents() const;
  const DeviceVector& getBetaCoefficentsOld() const;

  const DeviceMatrix& getWorkMatrixNxM() const;
  const DeviceVector& getWorkVectorNx1() const;
  const DeviceVector& getWorkVectorMx1() const;
  const DeviceMatrix& getUSVD() const;
  const DeviceMatrix& getVtSVD() const;
  const DeviceVector& getSigmaSVD() const;

private:
  void transferIntercept();

  const Configuration& configuration;
  const HostToDevice& hostToDevice;
  const int numberOfRows;
  const int numberOfPredictors;
  const KernelWrapper& kernelWrapper;
  DeviceMatrix* devicePredictors;
  DeviceVector* deviceOutcomes;
  int maxIterations;
  double convergenceThreshold;
  bool usingCovariates;
  PRECISION* devicePredictorsMemoryPointer;

  DeviceVector* betaCoefficentsDevice;
  DeviceVector* betaCoefficentsOldDevice;
  DeviceVector* probabilitesDevice;
  DeviceVector* scoresDevice;
  DeviceMatrix* informationMatrixDevice;
  DeviceMatrix* workMatrixNxMDevice;
  DeviceVector* workVectorNx1Device;
  DeviceVector* workVectorMx1Device;
  DeviceMatrix* uSVD;
  DeviceMatrix* vtSVD;
  DeviceVector* sigmaSVD;
};

} /* namespace LogisticRegression */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* LOGISTICREGRESSIONCONFIGURATION_H_ */