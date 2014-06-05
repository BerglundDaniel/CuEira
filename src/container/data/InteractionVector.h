#ifndef INTERACTIONVECTOR_H_
#define INTERACTIONVECTOR_H_

#include <HostVector.h>
#include <SNPVector.h>
#include <EnvironmentVector.h>
#include <Recode.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

namespace CuEira {
namespace Container {

/**
 * This class ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class InteractionVector {
public:
  InteractionVector(const EnvironmentVector& environmentVector, const SNPVector& snpVector);
  virtual ~InteractionVector();

  void recode();
  int getNumberOfIndividualsToInclude() const;
  const Container::HostVector& getRecodedData() const;

private:
  const EnvironmentVector& environmentVector;
  const SNPVector& snpVector;
  int numberOfIndividualsToInclude;
  Container::HostVector* interactionVector;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* INTERACTIONVECTOR_H_ */
