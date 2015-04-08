#ifndef MULTIPLICATIVEINTERACTIONMODEL_H_
#define MULTIPLICATIVEINTERACTIONMODEL_H_

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
class MultiplicativeInteractionModel: public InteractionModel<Vector> {
public:
  explicit MultiplicativeInteractionModel();
  virtual ~MultiplicativeInteractionModel();

  virtual void applyModel(SNPVector<Vector>& snpVector, EnvironmentVector<Vector>& environmentVector,
      InteractionVector<Vector>& interactionVector)=0;
};

} /* namespace CuEira */

#endif /* MULTIPLICATIVEINTERACTIONMODEL_H_ */
