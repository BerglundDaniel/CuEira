#ifndef INTERACTIONSTATISTICS_H_
#define INTERACTIONSTATISTICS_H_

#include <vector>
#include <math.h>
#include <iostream>
#include <ostream>

#include <DimensionMismatch.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <LogisticRegressionResult.h>

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class InteractionStatistics {
  friend std::ostream& operator<<(std::ostream& os, const InteractionStatistics& statistics);
public:
  InteractionStatistics(const Model::LogisticRegression::LogisticRegressionResult* logisticRegressionResult);
  virtual ~InteractionStatistics();

  double getReri() const;
  double getAp() const;
  const std::vector<double>& getOddsRatios() const;
  const std::vector<double>& getOddsRatiosLow() const;
  const std::vector<double>& getOddsRatiosHigh() const;

  InteractionStatistics(const InteractionStatistics&) = delete;
  InteractionStatistics(InteractionStatistics&&) = delete;
  InteractionStatistics& operator=(const InteractionStatistics&) = delete;
  InteractionStatistics& operator=(InteractionStatistics&&) = delete;

protected:
  virtual void toOstream(std::ostream& os) const;

private:
  std::vector<double>* calculateStandardError(const Container::HostMatrix& covarianceMatrix) const;
  double calculateReri(const std::vector<double>& oddsRatios) const;
  double calculateAp(double reri, PRECISION interactionBeta) const;
  std::vector<double>* calculateOddsRatios(const Container::HostVector& betaCoefficents) const;
  std::vector<double>* calculateOddsRatiosLow(const Container::HostVector& betaCoefficents,
      const std::vector<double>& standardError) const;
  std::vector<double>* calculateOddsRatiosHigh(const Container::HostVector& betaCoefficents,
      const std::vector<double>& standardError) const;

  const Model::LogisticRegression::LogisticRegressionResult* logisticRegressionResult;
  const Container::HostVector& betaCoefficents;
  std::vector<double>* standardError;
  std::vector<double>* oddsRatios;
  std::vector<double>* oddsRatiosLow;
  std::vector<double>* oddsRatiosHigh;
  double reri;
  double ap;
};

} /* namespace CuEira */

#endif /* INTERACTIONSTATISTICS_H_ */
