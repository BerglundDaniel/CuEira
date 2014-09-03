#ifndef CONTINGENCYTABLEFACTORY_H_
#define CONTINGENCYTABLEFACTORY_H_

#include <ContingencyTable.h>
#include <SNPVector.h>
#include <InteractionVector.h>
#include <EnvironmentVector.h>

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ContingencyTableFactory {
public:
  ContingencyTableFactory(const Container::HostVector& outcomes);
  virtual ~ContingencyTableFactory();

  virtual ContingencyTable* constructContingencyTable(const Container::SNPVector& snpVector,
      const Container::EnvironmentVector& environmentVector) const;

private:
  const Container::HostVector& outcomes;
  const int numberOfIndividualsToInclude;
};

} /* namespace CuEira */

#endif /* CONTINGENCYTABLEFACTORY_H_ */
