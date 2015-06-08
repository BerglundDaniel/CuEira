#include "CudaAlleleStatisticsFactory.h"

namespace CuEira {
namespace CUDA {

CudaAlleleStatisticsFactory::CudaAlleleStatisticsFactory(const KernelWrapper& kernelWrapper,
    const DeviceToHost& deviceToHost) :
    kernelWrapper(kernelWrapper), deviceToHost(deviceToHost) {

}

CudaAlleleStatisticsFactory::~CudaAlleleStatisticsFactory() {

}

std::vector<int>* CudaAlleleStatisticsFactory::getNumberOfAllelesPerGenotype(
    const Container::SNPVector<Container::DeviceVector>& snpVector,
    const Container::PhenotypeVector<Container::DeviceVector>& phenotypeVector) const {
  const Container::DeviceVector& snpData = snpVector.getOriginalSNPData();
  const Container::DeviceVector& phenotypeData = phenotypeVector.getPhenotypeData();

  const Container::DeviceMatrix* numberOfAllelesPerGenotypeBlockDevice =
      kernelWrapper.calculateNumberOfAllelesPerGenotype(snpData, phenotypeData);
  const Container::HostMatrix* numberOfAllelesPerGenotypeBlockHost = deviceToHost.transferMatrix(
      *numberOfAllelesPerGenotypeBlockDevice);

  std::vector<int>* numberOfAllelesPerGenotype = new std::vector<int>(6);

  const int numberOfBlocks = numberOfAllelesPerGenotypeBlockHost->getNumberOfRows();
  for(int b = 0; b < numberOfBlocks; ++b){
    for(int i = 0; i < 6; ++i){
      (*numberOfAllelesPerGenotype)[i] = (*numberOfAllelesPerGenotypeBlockHost)(b, i);
    }
  }

  delete numberOfAllelesPerGenotypeBlockDevice;
  delete numberOfAllelesPerGenotypeBlockHost;

  return numberOfAllelesPerGenotype;
}

} /* namespace CUDA */
} /* namespace CuEira */