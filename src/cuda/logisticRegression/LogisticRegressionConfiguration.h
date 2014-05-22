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
      const HostVector& outcomes);

  /**
   * This
   */
  LogisticRegressionConfiguration(const Configuration& configuration, const HostToDevice& hostToDevice,
      const HostVector& outcomes, const HostMatrix& covariates);

  /**
   * This
   */
  virtual ~LogisticRegressionConfiguration();

  void setEnvironmentFactor(const HostVector& environmentData);
  void setSNP(const HostVector& snpData);

  int getNumberOfRows() const;
  int getNumberOfPredictors() const;
  int getNumberOfMaxIterations() const;
  double getConvergenceThreshold() const;
  const DeviceMatrix& getPredictors() const;
  const DeviceVector& getOutcomes() const;

private:
  const Configuration& configuration;
  const HostToDevice& hostToDevice;
  const int numberOfRows;
  const int numberOfPredictors;
  int maxIterations;
  double convergenceThreshold;
  bool usingCovariates;
  DeviceMatrix* devicePredictors;
  DeviceMatrix* deviceOutcomes;
};

} /* namespace LogisticRegression */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* LOGISTICREGRESSIONCONFIGURATION_H_ */
