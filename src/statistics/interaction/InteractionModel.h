#ifndef INTERACTIONMODEL_H_
#define INTERACTIONMODEL_H_

#include <EnvironmentVector.h>
#include <InteractionVector.h>
#include <SNPVector.h>

namespace CuEira {

using namespace CuEira::Container;

/*
 * This class ...
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Vector>
class InteractionModel {
public:
  explicit InteractionModel();
  virtual ~InteractionModel();

  virtual void applyModel(SNPVector<Vector>& snpVector, EnvironmentVector<Vector>& environmentVector,
      InteractionVector<Vector>& interactionVector)=0;
};

} /* namespace CuEira */

#endif /* INTERACTIONMODEL_H_ */
