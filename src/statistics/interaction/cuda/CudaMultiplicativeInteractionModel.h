#ifndef CUDAMULTIPLICATIVEINTERACTIONMODEL_H_
#define CUDAMULTIPLICATIVEINTERACTIONMODEL_H_

#include <MultiplicativeInteractionModel.h>
#include <EnvironmentVector.h>
#include <InteractionVector.h>
#include <SNPVector.h>
#include <DeviceVector.h>
#include <KernelWrapper.h>

namespace CuEira {
namespace CUDA {

using namespace CuEira::Container;

/*
 * This class ...
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaMultiplicativeInteractionModel: public MultiplicativeInteractionModel<DeviceVector> {
public:
  explicit CudaMultiplicativeInteractionModel(const KernelWrapper& kernelWrapper);
  virtual ~CudaMultiplicativeInteractionModel();

  virtual void applyModel(SNPVector<DeviceVector>& snpVector, EnvironmentVector<DeviceVector>& environmentVector,
      InteractionVector<DeviceVector>& interactionVector);

protected:
  const KernelWrapper& kernelWrapper;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* CUDAMULTIPLICATIVEINTERACTIONMODEL_H_ */
