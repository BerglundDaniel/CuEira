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
#include <OddsRatioStatistics.h>

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class InteractionStatistics: public OddsRatioStatistics {
  friend std::ostream& operator<<(std::ostream& os, const InteractionStatistics& statistics);
public:
  explicit InteractionStatistics(const Model::LogisticRegression::LogisticRegressionResult* logisticRegressionResult);
  virtual ~InteractionStatistics();

  virtual double getReri() const;
  virtual double getAp() const;

  InteractionStatistics(const InteractionStatistics&) = delete;
  InteractionStatistics(InteractionStatistics&&) = delete;
  InteractionStatistics& operator=(const InteractionStatistics&) = delete;
  InteractionStatistics& operator=(InteractionStatistics&&) = delete;

protected:
  InteractionStatistics(); //For the mock
  virtual void toOstream(std::ostream& os) const;

private:
  double calculateReri(const std::vector<double>& oddsRatios) const;
  double calculateAp(double reri, PRECISION interactionBeta) const;

  double reri;
  double ap;
};

} /* namespace CuEira */

#endif /* INTERACTIONSTATISTICS_H_ */
