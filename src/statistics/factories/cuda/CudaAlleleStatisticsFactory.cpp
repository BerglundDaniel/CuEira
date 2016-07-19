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
  std::vector<int>* numberOfAllelesPerGenotype = new std::vector<int>(6);

  const Container::DeviceMatrix* numberOfAllelesPerGenotypeBlockDevice = Kernel::calculateNumberOfAllelesPerGenotype(
      stream, snpData, phenotypeData);

  Kernel: absoluteSumToHost(stream, (*numberOfAllelesPerGenotypeBlockDevice)(0), &((*numberOfAllelesPerGenotype)[0]));
  Kernel: absoluteSumToHost(stream, (*numberOfAllelesPerGenotypeBlockDevice)(1), &((*numberOfAllelesPerGenotype)[1]));
  Kernel: absoluteSumToHost(stream, (*numberOfAllelesPerGenotypeBlockDevice)(2), &((*numberOfAllelesPerGenotype)[2]));
  Kernel: absoluteSumToHost(stream, (*numberOfAllelesPerGenotypeBlockDevice)(3), &((*numberOfAllelesPerGenotype)[3]));
  Kernel: absoluteSumToHost(stream, (*numberOfAllelesPerGenotypeBlockDevice)(4), &((*numberOfAllelesPerGenotype)[4]));
  Kernel: absoluteSumToHost(stream, (*numberOfAllelesPerGenotypeBlockDevice)(5), &((*numberOfAllelesPerGenotype)[5]));

  delete numberOfAllelesPerGenotypeBlockDevice;

  return numberOfAllelesPerGenotype;
}

} /* namespace CUDA */
} /* namespace CuEira */
