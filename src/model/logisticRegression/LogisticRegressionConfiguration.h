#ifndef LOGISTICREGRESSIONCONFIGURATION_H_
#define LOGISTICREGRESSIONCONFIGURATION_H_

#include <Configuration.h>
#include <ModelConfiguration.h>
#include <MKLWrapper.h>
#include <HostMatrix.h>
#include <HostVector.h>

namespace CuEira {
namespace Model {
namespace LogisticRegression {

using namespace CuEira::Container;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LogisticRegressionConfiguration: public ModelConfiguration {
public:
  LogisticRegressionConfiguration(const Configuration& configuration, bool usingCovariates, const int numberOfRows,
      const int numberOfPredictors);
  virtual ~LogisticRegressionConfiguration();

  virtual int getNumberOfRows() const;
  virtual int getNumberOfPredictors() const;
  virtual int getNumberOfMaxIterations() const;
  virtual double getConvergenceThreshold() const;
  virtual HostVector& getScoresHost()=0;

  virtual void setEnvironmentFactor(const HostVector& environmentData)=0;
  virtual void setSNP(const HostVector& snpData)=0;
  virtual void setInteraction(const HostVector& interactionVector)=0;

  LogisticRegressionConfiguration(const LogisticRegressionConfiguration&) = delete;
  LogisticRegressionConfiguration(LogisticRegressionConfiguration&&) = delete;
  LogisticRegressionConfiguration& operator=(const LogisticRegressionConfiguration&) = delete;
  LogisticRegressionConfiguration& operator=(LogisticRegressionConfiguration&&) = delete;

protected:
  void setDefaultBeta(HostVector& beta);

  int maxIterations;
  double convergenceThreshold;
  const int numberOfPredictors;
  const int numberOfRows;
  const bool usingCovariates;
};

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif /* LOGISTICREGRESSIONCONFIGURATION_H_ */
