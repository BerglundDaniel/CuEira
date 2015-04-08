#ifndef ADDITIVEINTERACTIONMODEL_H_
#define ADDITIVEINTERACTIONMODEL_H_

#include <InteractionModel.h>
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
class AdditiveInteractionModel: public InteractionModel<Vector> {
public:
  explicit AdditiveInteractionModel();
  virtual ~AdditiveInteractionModel();

  virtual void applyModel(SNPVector<Vector>& snpVector, EnvironmentVector<Vector>& environmentVector,
      InteractionVector<Vector>& interactionVector)=0;
};

} /* namespace CuEira */

#endif /* ADDITIVEINTERACTIONMODEL_H_ */
