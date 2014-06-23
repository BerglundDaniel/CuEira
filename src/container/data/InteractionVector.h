#ifndef INTERACTIONVECTOR_H_
#define INTERACTIONVECTOR_H_

#include <HostVector.h>
#include <SNPVector.h>
#include <EnvironmentVector.h>
#include <Recode.h>
#include <InvalidState.h>

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
  InteractionVector(const EnvironmentVector& environmentVector);
  virtual ~InteractionVector();

  virtual void recode(const SNPVector& snpVector);
  virtual int getNumberOfIndividualsToInclude() const;
  virtual const Container::HostVector& getRecodedData() const;

protected:
  InteractionVector();

private:
  enum State {
    NOT_INITIALISED, INITIALISED
  };

  State state;
  const EnvironmentVector* environmentVector;
  int numberOfIndividualsToInclude;
  Container::HostVector* interactionVector;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* INTERACTIONVECTOR_H_ */
