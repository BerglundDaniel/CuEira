#ifndef CUDASNPVECTORFACTORY_H_
#define CUDASNPVECTORFACTORY_H_

#include <SNPVectorFactory.h>
#include <DeviceVector.h>
#include <SNPVector.h>
#include <CudaSNPVector.h>
#include <KernelWrapper.h>
#include <HostToDevice.h>
#include <Configuration.h>

namespace CuEira {
namespace Container {
namespace CUDA {

/*
 * This class ...
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaSNPVectorFactory: public SNPVectorFactory {
public:
  explicit CudaSNPVectorFactory(const Configuration& configuration, const HostToDevice& hostToDevice,
      const KernelWrapper& kernelWrapper);
  virtual ~CudaSNPVectorFactory();

  virtual SNPVector<DeviceVector>* constructSNPVector(SNP& snp, const HostVector* originalSNPData,
      const std::set<int>* snpMissingData) const;

  CudaSNPVectorFactory(const CudaSNPVectorFactory&) = delete;
  CudaSNPVectorFactory(CudaSNPVectorFactory&&) = delete;
  CudaSNPVectorFactory& operator=(const CudaSNPVectorFactory&) = delete;
  CudaSNPVectorFactory& operator=(CudaSNPVectorFactory&&) = delete;

private:
  const HostToDevice& hostToDevice;
  const KernelWrapper& kernelWrapper;
};

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CUDASNPVECTORFACTORY_H_ */
