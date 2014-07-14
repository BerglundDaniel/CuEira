#ifndef STATISTICSFACTORY_H_
#define STATISTICSFACTORY_H_

#include <Statistics.h>
#include <LogisticRegressionResult.h>

namespace CuEira {

/**
 * This is class is responsible for creating instances of Statistics
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class StatisticsFactory {
public:
  StatisticsFactory();
  virtual ~StatisticsFactory();

  virtual Statistics* constructStatistics(const Model::LogisticRegression::LogisticRegressionResult* logisticRegressionResult) const;
};

} /* namespace CuEira */

#endif /* STATISTICSFACTORY_H_ */
