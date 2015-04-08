#ifndef CUDAMISSINGDATAHANDLER_H_
#define CUDAMISSINGDATAHANDLER_H_

#include <MissingDataHandler.h>
#include <DeviceVector.h>
#include <HostToDevice.h>
#include <KernelWrapper.h>

namespace CuEira {
namespace CUDA {

/*
 * This class ...
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaMissingDataHandler: public MissingDataHandler<Container::DeviceVector> {
public:
  explicit CudaMissingDataHandler(const int numberOfIndividualsTotal, const HostToDevice& hostToDevice,
      const KernelWrapper& kernelWrapper);
  virtual ~CudaMissingDataHandler();

  virtual void setMissing(const std::set<int>& snpPersonsToSkip);
  virtual void copyNonMissing(const Container::DeviceVector& fromVector, Container::DeviceVector& toVector) const;

protected:
  const HostToDevice& hostToDevice;
  const KernelWrapper& kernelWrapper;
  Container::DeviceVector* indexesToCopyDevice;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* CUDAMISSINGDATAHANDLER_H_ */