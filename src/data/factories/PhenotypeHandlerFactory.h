#ifndef PHENOTYPEHANDLERFACTORY_H_
#define PHENOTYPEHANDLERFACTORY_H_

#include <vector>

#include <Person.h>
#include <Phenotype.h>
#include <PersonHandler.h>
#include <PhenotypeHandler.h>

#ifdef CPU
#include <RegularHostVector.h>
#include <CpuPhenotypeHandler.h>
#else
#include <CudaPhenotypeHandler.h>
#include <PinnedHostVector.h>
#endif

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PhenotypeHandlerFactory {
public:
  explicit PhenotypeHandlerFactory();
  virtual ~PhenotypeHandlerFactory();

  virtual PhenotypeHandler* constructPhenotypeHandler(const PersonHandler& personHandler) const;
};

} /* namespace CuEira */

#endif /* PHENOTYPEHANDLERFACTORY_H_ */
