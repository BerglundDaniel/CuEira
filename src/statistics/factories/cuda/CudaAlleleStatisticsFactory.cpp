#include "CudaAlleleStatisticsFactory.h"

namespace CuEira {
namespace CUDA {

CudaAlleleStatisticsFactory::CudaAlleleStatisticsFactory(const Stream& stream) :
    stream(stream){

}

CudaAlleleStatisticsFactory::~CudaAlleleStatisticsFactory(){

}

std::vector<int>* CudaAlleleStatisticsFactory::getNumberOfAllelesPerGenotype(
    const Container::SNPVector<Container::DeviceVector>& snpVector,
    const Container::PhenotypeVector<Container::DeviceVector>& phenotypeVector) const{
  const Container::DeviceVector& snpData = snpVector.getOriginalSNPData();
  const Container::DeviceVector& phenotypeData = phenotypeVector.getPhenotypeData();

  const Container::DeviceMatrix* numberOfAllelesPerGenotypeBlockDevice = Kernel::calculateNumberOfAllelesPerGenotype(
      stream, snpData, phenotypeData);
  const Container::HostMatrix* numberOfAllelesPerGenotypeBlockHost = transferMatrix(stream,
      *numberOfAllelesPerGenotypeBlockDevice);

  std::vector<int>* numberOfAllelesPerGenotype = new std::vector<int>(6);

  const int numberOfBlocks = numberOfAllelesPerGenotypeBlockHost->getNumberOfRows();
  for(int b = 0; b < numberOfBlocks; ++b){
    (*numberOfAllelesPerGenotype)[0] = (*numberOfAllelesPerGenotypeBlockHost)(b, 0);
    (*numberOfAllelesPerGenotype)[1] = (*numberOfAllelesPerGenotypeBlockHost)(b, 1);
    (*numberOfAllelesPerGenotype)[2] = (*numberOfAllelesPerGenotypeBlockHost)(b, 2);
    (*numberOfAllelesPerGenotype)[3] = (*numberOfAllelesPerGenotypeBlockHost)(b, 3);
    (*numberOfAllelesPerGenotype)[4] = (*numberOfAllelesPerGenotypeBlockHost)(b, 4);
    (*numberOfAllelesPerGenotype)[5] = (*numberOfAllelesPerGenotypeBlockHost)(b, 5);
  }

  delete numberOfAllelesPerGenotypeBlockDevice;
  delete numberOfAllelesPerGenotypeBlockHost;

  return numberOfAllelesPerGenotype;
}

} /* namespace CUDA */
} /* namespace CuEira */
