#ifndef CPULOGISTICREGRESSIONCONFIGURATION_H_
#define CPULOGISTICREGRESSIONCONFIGURATION_H_

#include <Configuration.h>
#include <LogisticRegressionConfiguration.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <RegularHostMatrix.h>
#include <RegularHostVector.h>
#include <MKLWrapper.h>

namespace CuEira {
namespace Model {
namespace LogisticRegression {
namespace CPU {

using namespace CuEira::Container;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuLogisticRegressionConfiguration: public LogisticRegressionConfiguration {
public:
  CpuLogisticRegressionConfiguration(const Configuration& configuration, const HostVector& outcomes);
  CpuLogisticRegressionConfiguration(const Configuration& configuration, const HostVector& outcomes,
      const HostMatrix& covariates);
  virtual ~CpuLogisticRegressionConfiguration();

  virtual void setEnvironmentFactor(const HostVector& environmentData);
  virtual void setSNP(const HostVector& snpData);
  virtual void setInteraction(const HostVector& interactionVector);

  virtual HostVector& getProbabilites();
  virtual HostVector& getScoresHost();

  virtual const HostMatrix& getPredictors() const;
  virtual const HostVector& getOutcomes() const;

  virtual HostMatrix& getWorkMatrixNxM();
  virtual HostVector& getWorkVectorNx1();

  virtual const HostVector& getDefaultBetaCoefficents() const;

  CpuLogisticRegressionConfiguration(const CpuLogisticRegressionConfiguration&) = delete;
  CpuLogisticRegressionConfiguration(CpuLogisticRegressionConfiguration&&) = delete;
  CpuLogisticRegressionConfiguration& operator=(const CpuLogisticRegressionConfiguration&) = delete;
  CpuLogisticRegressionConfiguration& operator=(CpuLogisticRegressionConfiguration&&) = delete;

protected:
  CpuLogisticRegressionConfiguration(const Configuration& configuration); //For the mock

private:
  void setIntercept();
  void setCovariates(const HostMatrix& covariates);

  const HostVector* outcomes;
  HostMatrix* predictors;
  PRECISION* predictorsMemoryPointer;
  Container::HostVector* probabilites;
  Container::HostMatrix* workMatrixNxM;
  Container::HostVector* workVectorNx1;
  Container::HostVector* defaultBetaCoefficents;
  HostVector* scoresHost;
};

} /* namespace CPU */
} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif /* CPULOGISTICREGRESSIONCONFIGURATION_H_ */
