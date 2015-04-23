#ifndef ODDSRATIOSTATISTICS_H_
#define ODDSRATIOSTATISTICS_H_

#include <vector>
#include <math.h>
#include <iostream>
#include <ostream>

#include <ModelStatistics.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <LogisticRegressionResult.h>

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class OddsRatioStatistics:public ModelStatistics {
  friend std::ostream& operator<<(std::ostream& os, const OddsRatioStatistics& oddsRatioStatistics);
public:
  explicit OddsRatioStatistics(const Model::LogisticRegression::LogisticRegressionResult* logisticRegressionResult);
  virtual ~OddsRatioStatistics();

  virtual const std::vector<double>& getOddsRatios() const;
  virtual const std::vector<double>& getOddsRatiosLow() const;
  virtual const std::vector<double>& getOddsRatiosHigh() const;

  OddsRatioStatistics(const OddsRatioStatistics&) = delete;
  OddsRatioStatistics(OddsRatioStatistics&&) = delete;
  OddsRatioStatistics& operator=(const OddsRatioStatistics&) = delete;
  OddsRatioStatistics& operator=(OddsRatioStatistics&&) = delete;

  const int numberOfIterations;

protected:
  OddsRatioStatistics(); //For the mock
  virtual void toOstream(std::ostream& os) const;

  const Model::LogisticRegression::LogisticRegressionResult* logisticRegressionResult;
  const Container::HostVector* betaCoefficents;
  std::vector<double>* standardError;
  std::vector<double>* oddsRatios;
  std::vector<double>* oddsRatiosLow;
  std::vector<double>* oddsRatiosHigh;

private:
  std::vector<double>* calculateStandardError(const Container::HostMatrix& covarianceMatrix) const;
  std::vector<double>* calculateOddsRatios(const Container::HostVector& betaCoefficents) const;
  std::vector<double>* calculateOddsRatiosLow(const Container::HostVector& betaCoefficents,
      const std::vector<double>& standardError) const;
  std::vector<double>* calculateOddsRatiosHigh(const Container::HostVector& betaCoefficents,
      const std::vector<double>& standardError) const;
};

} /* namespace CuEira */

#endif /* ODDSRATIOSTATISTICS_H_ */
