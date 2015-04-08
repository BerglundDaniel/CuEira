#ifndef ALLELESTATISTICSFACTORY_H_
#define ALLELESTATISTICSFACTORY_H_

#include <vector>

#include <AlleleStatistics.h>

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class AlleleStatisticsFactory {
public:
  AlleleStatisticsFactory();
  virtual ~AlleleStatisticsFactory();

  virtual AlleleStatistics* constructAlleleStatistics(const std::vector<int>* numberOfAlleles) const;

private:
  std::vector<double>* convertAlleleNumbersToFrequencies(const std::vector<int>& numberOfAlleles) const;
};

} /* namespace CuEira */

#endif /* ALLELESTATISTICSFACTORY_H_ */
