#ifndef ALLELESTATISTICSFACTORY_H_
#define ALLELESTATISTICSFACTORY_H_

#include <vector>

#include <AlleleStatistics.h>
#include <SNPVector.h>
#include <PhenotypeVector.h>

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Vector>
class AlleleStatisticsFactory {
public:
  explicit AlleleStatisticsFactory();
  virtual ~AlleleStatisticsFactory();

  virtual AlleleStatistics* constructAlleleStatistics(const Container::SNPVector<Vector>& snpVector,
      const Container::PhenotypeVector<Vector>& phenotypeVector) const;

protected:
  virtual std::vector<int>* getNumberOfAllelesPerGenotype(const Container::SNPVector<Vector>& snpVector,
      const Container::PhenotypeVector<Vector>& phenotypeVector) const=0;
  std::vector<double>* convertAlleleNumbersToFrequencies(const std::vector<int>& numberOfAlleles) const;
};

} /* namespace CuEira */

#endif /* ALLELESTATISTICSFACTORY_H_ */
