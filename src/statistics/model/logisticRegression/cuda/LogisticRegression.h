#ifndef LOGISTICREGRESSION_H_
#define LOGISTICREGRESSION_H_

#include <mkl.h>

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
#include <InvalidState.h>

namespace CuEira {
namespace Model {
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
  LogisticRegression(LogisticRegressionConfiguration& lrConfiguration, const HostToDevice& hostToDevice, const DeviceToHost& deviceToHost);
  virtual ~LogisticRegression();

  /**
   * Get the vector containing the beta coefficients.
   */
  const HostVector& getBeta() const;

  /**
   * Get the matrix containing the covariances
   */
  const HostMatrix& getCovarianceMatrix() const;

  /**
   * Get the information matrix.
   */
  const HostMatrix& getInformationMatrix() const;

  /**
   * Get the number of iterations it took for the model to converge.
   */
  int getNumberOfIterations() const;

  /**
   * Get the loglikelihood score.
   */
  PRECISION getLogLikelihood() const;

private:
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
  Container::DeviceMatrix& informationMatrixDevice;
  Container::DeviceVector& betaCoefficentsDevice;

  Container::HostMatrix* informationMatrixHost;
  Container::HostVector* scoresHost;
  Container::HostVector* betaCoefficentsHost;
};

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif /* LOGISTICREGRESSION_H_ */
