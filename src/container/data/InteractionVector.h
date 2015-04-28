#ifndef INTERACTIONVECTOR_H_
#define INTERACTIONVECTOR_H_

#include <InvalidState.h>
#include <InvalidArgument.h>

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
  explicit InteractionVector(int numberOfIndividualsTotal);
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
  const int numberOfIndividualsTotal;
  int numberOfIndividualsToInclude;
  Vector* interactionExMissing;
  bool initialised;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* INTERACTIONVECTOR_H_ */
