#ifndef CPUCpuPhenotypeHandlerFactory_H_
#define CPUCpuPhenotypeHandlerFactory_H_

#include <PhenotypeHandlerFactory.h>
#include <Person.h>
#include <Phenotype.h>
#include <PersonHandler.h>
#include <PhenotypeHandler.h>
#include <RegularHostVector.h>

namespace CuEira {
namespace CPU {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuPhenotypeHandlerFactory: public PhenotypeHandlerFactory<Container::RegularHostVector> {
public:
  explicit CpuPhenotypeHandlerFactory();
  virtual ~CpuPhenotypeHandlerFactory();

  virtual PhenotypeHandler<Container::HostVector>* constructPhenotypeHandler(const PersonHandler& personHandler) const;
};

} /* namespace CPU */
} /* namespace CuEira */

#endif /* CPUCpuPhenotypeHandlerFactory_H_ */
