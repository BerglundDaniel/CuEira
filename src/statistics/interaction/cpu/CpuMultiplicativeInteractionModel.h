#ifndef CPUMULTIPLICATIVEINTERACTIONMODEL_H_
#define CPUMULTIPLICATIVEINTERACTIONMODEL_H_

#include <MultiplicativeInteractionModel.h>
#include <EnvironmentVector.h>
#include <InteractionVector.h>
#include <SNPVector.h>
#include <RegularHostVector.h>
#include <MKLWrapper.h>

namespace CuEira {
namespace CPU {

using namespace CuEira::Container;

/*
 * This class ...
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuMultiplicativeInteractionModel: public MultiplicativeInteractionModel<RegularHostVector> {
public:
  explicit CpuMultiplicativeInteractionModel(const MKLWrapper& mklWrapper);
  virtual ~CpuMultiplicativeInteractionModel();

  virtual void applyModel(SNPVector<RegularHostVector>& snpVector,
      EnvironmentVector<RegularHostVector>& environmentVector, InteractionVector<RegularHostVector>& interactionVector);

protected:
  const MKLWrapper& mklWrapper;
};

} /* namespace CPU */
} /* namespace CuEira */

#endif /* CPUMULTIPLICATIVEINTERACTIONMODEL_H_ */
