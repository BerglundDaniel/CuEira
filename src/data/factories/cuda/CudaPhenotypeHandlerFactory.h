#ifndef CUDAPHENOTYPEHANDLERFACTORY_H_
#define CUDAPHENOTYPEHANDLERFACTORY_H_

#include <PhenotypeHandlerFactory.h>
#include <Person.h>
#include <Phenotype.h>
#include <PersonHandler.h>
#include <PhenotypeHandler.h>
#include <PinnedHostVector.h>
#include <DeviceVector.h>
#include <HostToDevice.h>

namespace CuEira {
namespace CUDA {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaPhenotypeHandlerFactory: public PhenotypeHandlerFactory<Container::PinnedHostVector> {
public:
  explicit CudaPhenotypeHandlerFactory();
  virtual ~CudaPhenotypeHandlerFactory();

  virtual PhenotypeHandler<Container::DeviceVector>* constructPhenotypeHandler(const PersonHandler& personHandler,
      const HostToDevice& hostToDevice) const;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* CUDAPHENOTYPEHANDLERFACTORY_H_ */
