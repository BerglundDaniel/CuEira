#ifndef PHENOTYPEHANDLERFACTORY_H_
#define PHENOTYPEHANDLERFACTORY_H_

#include <vector>

#include <PersonHandlerLocked.h>
#include <Person.h>
#include <Phenotype.h>

namespace CuEira {

/*
 * This class....
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Vector>
class PhenotypeHandlerFactory {
public:
  virtual ~PhenotypeHandlerFactory();

  PhenotypeHandlerFactory(const PhenotypeHandlerFactory&) = delete;
  PhenotypeHandlerFactory(PhenotypeHandlerFactory&&) = delete;
  PhenotypeHandlerFactory& operator=(const PhenotypeHandlerFactory&) = delete;
  PhenotypeHandlerFactory& operator=(PhenotypeHandlerFactory&&) = delete;

protected:
  explicit PhenotypeHandlerFactory();

  virtual Vector* createVectorOfPhenotypes(const PersonHandlerLocked& personHandlerLocked) const;
};

} /* namespace CuEira */

#endif /* PHENOTYPEHANDLERFACTORY_H_ */
