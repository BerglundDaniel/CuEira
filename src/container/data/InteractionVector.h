#ifndef INTERACTIONVECTOR_H_
#define INTERACTIONVECTOR_H_

#include <HostVector.h>
#include <SNPVector.h>
#include <Recode.h>
#include <InvalidState.h>

#ifdef CPU
#include <RegularHostVector.h>
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
template<typename Vector>
class InteractionVector {
public:
  explicit InteractionVector();
  virtual ~InteractionVector();

  virtual int getNumberOfIndividualsToInclude() const;
  virtual const Vector& getInteractionData() const;
  virtual Vector& getInteractionData();
  virtual void updateSize(int size);

  InteractionVector(const InteractionVector&) = delete;
  InteractionVector(InteractionVector&&) = delete;
  InteractionVector& operator=(const InteractionVector&) = delete;
  InteractionVector& operator=(InteractionVector&&) = delete;

private:
  Vector* interactionExMissing;
  int numberOfIndividualsToInclude;
  bool initialised;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* INTERACTIONVECTOR_H_ */
