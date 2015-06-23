#ifndef CPUCONTINGENCYTABLEFACTORY_H_
#define CPUCONTINGENCYTABLEFACTORY_H_

#include <vector>

#include <ContingencyTable.h>
#include <SNPVector.h>
#include <PhenotypeVector.h>
#include <EnvironmentVector.h>
#include <ContingencyTableFactory.h>
#include <RegularHostVector.h>

namespace CuEira {
namespace CPU {

/*
 * This class....
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuContingencyTableFactory: public ContingencyTableFactory<Container::RegularHostVector> {
public:
  explicit CpuContingencyTableFactory();
  virtual ~CpuContingencyTableFactory();

  virtual const ContingencyTable* constructContingencyTable(
      const Container::SNPVector<Container::RegularHostVector>& snpVector,
      const Container::EnvironmentVector<Container::RegularHostVector>& environmentVector,
      const Container::PhenotypeVector<Container::RegularHostVector>& phenotypeVector) const;

  CpuContingencyTableFactory(const CpuContingencyTableFactory&) = delete;
  CpuContingencyTableFactory(CpuContingencyTableFactory&&) = delete;
  CpuContingencyTableFactory& operator=(const CpuContingencyTableFactory&) = delete;
  CpuContingencyTableFactory& operator=(CpuContingencyTableFactory&&) = delete;

private:

};

} /* namespace CPU */
} /* namespace CuEira */

#endif /* CPUCONTINGENCYTABLEFACTORY_H_ */
