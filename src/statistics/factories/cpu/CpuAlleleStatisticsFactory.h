#ifndef CPUALLELESTATISTICSFACTORY_H_
#define CPUALLELESTATISTICSFACTORY_H_

#include <AlleleStatisticsFactory.h>
#include <RegularHostVector.h>
#include <DimensionMismatch.h>
#include <SNPVector.h>
#include <PhenotypeVector.h>

namespace CuEira {
namespace CPU {

/*
 * This class ...
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuAlleleStatisticsFactory: public CuEira::AlleleStatisticsFactory<Container::RegularHostVector> {
public:
  explicit CpuAlleleStatisticsFactory();
  virtual ~CpuAlleleStatisticsFactory();

protected:
  virtual std::vector<int>* getNumberOfAllelesPerGenotype(
      const Container::SNPVector<Container::RegularHostVector>& snpVector,
      const Container::PhenotypeVector<Container::RegularHostVector>& phenotypeVector) const;
};

} /* namespace CPU */
} /* namespace CuEira */

#endif /* CPUALLELESTATISTICSFACTORY_H_ */
