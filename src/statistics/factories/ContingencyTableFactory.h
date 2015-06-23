#ifndef CONTINGENCYTABLEFACTORY_H_
#define CONTINGENCYTABLEFACTORY_H_

#include <ContingencyTable.h>
#include <SNPVector.h>
#include <PhenotypeVector.h>
#include <EnvironmentVector.h>

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Vector>
class ContingencyTableFactory {
public:
  ContingencyTableFactory();
  virtual ~ContingencyTableFactory();

  virtual const ContingencyTable* constructContingencyTable(const Container::SNPVector<Vector>& snpVector,
      const Container::EnvironmentVector<Vector>& environmentVector,
      const Container::PhenotypeVector<Vector>& phenotypeVector) const=0;

protected:
  const static int tableSize = 8;
};

} /* namespace CuEira */

#endif /* CONTINGENCYTABLEFACTORY_H_ */
