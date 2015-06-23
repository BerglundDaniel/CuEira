#ifndef CUDACONTINGENCYTABLEFACTORY_H_
#define CUDACONTINGENCYTABLEFACTORY_H_

#include <vector>

#include <ContingencyTable.h>
#include <SNPVector.h>
#include <PhenotypeVector.h>
#include <EnvironmentVector.h>
#include <ContingencyTableFactory.h>
#include <DeviceVector.h>
#include <HostToDevice.h>
#include <KernelWrapper.h>
#include <Stream.h>

namespace CuEira {
namespace CUDA {

/*
 * This class....
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaContingencyTableFactory: public ContingencyTableFactory<Container::DeviceVector> {
public:
  explicit CudaContingencyTableFactory();
  virtual ~CudaContingencyTableFactory();

  virtual const ContingencyTable* constructContingencyTable(
      const Container::SNPVector<Container::DeviceVector>& snpVector,
      const Container::EnvironmentVector<Container::DeviceVector>& environmentVector,
      const Container::PhenotypeVector<Container::DeviceVector>& phenotypeVector) const;

  CudaContingencyTableFactory(const CudaContingencyTableFactory&) = delete;
  CudaContingencyTableFactory(CudaContingencyTableFactory&&) = delete;
  CudaContingencyTableFactory& operator=(const CudaContingencyTableFactory&) = delete;
  CudaContingencyTableFactory& operator=(CudaContingencyTableFactory&&) = delete;

private:

};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* CUDACONTINGENCYTABLEFACTORY_H_ */
