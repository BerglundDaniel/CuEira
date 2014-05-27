#ifndef LOGISTICREGRESSION_H_
#define LOGISTICREGRESSION_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <CudaAdapter.cu>
#include <KernelWrapper.h>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <CudaException.h>
#include <CublasException.h>
#include <DimensionMismatch.h>
#include <LogisticRegressionConfiguration.h>

namespace CuEira {
namespace CUDA {
namespace LogisticRegression {

using namespace CuEira::Container;
using namespace CuEira::CUDA;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LogisticRegression {
public:
  LogisticRegression(LogisticRegressionConfiguration& lrConfiguration);
  virtual ~LogisticRegression();

  /**
   * Get the vector containing the beta coefficients.
   */
  const DeviceVector& getBeta() const;

  /**
   * Get the matrix containing the covariances
   */
  const DeviceVector& getCovarianceMatrix() const;

  /**
   * Get the information matrix.
   */
  const DeviceMatrix& getInformationMatrix() const;

  /**
   * Get the number of iterations it took for the model to converge.
   */
  int getNumberOfIterations() const;

  /**
   * Get the loglikelihood score.
   */
  PRECISION getLogLikelihood() const;

private:
  LogisticRegressionConfiguration& lrConfiguration;
  const int numberOfRows;
  const int numberOfPredictors;
  const int maxIterations;
  const double convergenceThreshold;
  const KernelWrapper& kernelWrapper;
  int iterationNumber;
  PRECISION* logLikelihood;
  Container::DeviceMatrix& informationMatrixDevice;
  Container::DeviceVector& betaCoefficentsDevice;
  Container::DeviceMatrix& inverseInformationMatrixDevice;
};

} /* namespace LogisticRegression */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* LOGISTICREGRESSION_H_ */
