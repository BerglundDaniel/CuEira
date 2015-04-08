#ifndef CPUADDITIVEINTERACTIONMODEL_H_
#define CPUADDITIVEINTERACTIONMODEL_H_

#include <AdditiveInteractionModel.h>
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
class CpuAdditiveInteractionModel: public AdditiveInteractionModel<RegularHostVector> {
public:
  explicit CpuAdditiveInteractionModel(const MKLWrapper& mklWrapper);
  virtual ~CpuAdditiveInteractionModel();

  virtual void applyModel(SNPVector<RegularHostVector>& snpVector,
      EnvironmentVector<RegularHostVector>& environmentVector, InteractionVector<RegularHostVector>& interactionVector);

protected:
  const MKLWrapper& mklWrapper;
};

} /* namespace CPU */
} /* namespace CuEira */

#endif /* CPUADDITIVEINTERACTIONMODEL_H_ */
