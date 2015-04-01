#ifndef CUDASNPVECTOR_H_
#define CUDASNPVECTOR_H_

#include <set>

#include <SNPVector.h>
#include <DeviceVector.h>
#include <SNP.h>
#include <GeneticModel.h>
#include <Recode.h>
#include <KernelWrapper.h>

namespace CuEira {
namespace Container {
namespace CUDA {

using namespace CuEira::CUDA;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaSNPVector: public SNPVector<DeviceVector> {
public:
  explicit CudaSNPVector(SNP& snp, GeneticModel geneticModel, const DeviceVector* snpOrgExMissing,
      const std::set<int>* snpMissingData, const KernelWrapper& kernelWrapper);
  virtual ~CudaSNPVector();

protected:
  virtual void doRecode(int snpToRisk[3]);

  const KernelWrapper& kernelWrapper;
};

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CUDASNPVECTOR_H_ */
