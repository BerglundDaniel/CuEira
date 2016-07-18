#include "CpuAlleleStatisticsFactory.h"

namespace CuEira {
namespace CPU {

CpuAlleleStatisticsFactory::CpuAlleleStatisticsFactory() {

}

CpuAlleleStatisticsFactory::~CpuAlleleStatisticsFactory() {

}

std::vector<int>* CpuAlleleStatisticsFactory::getNumberOfAllelesPerGenotype(
    const Container::SNPVector<Container::RegularHostVector>& snpVector,
    const Container::PhenotypeVector<Container::RegularHostVector>& phenotypeVector) const {
#ifdef DEBUG
  if(snpVector.getNumberOfIndividualsToInclude() != phenotypeVector.getNumberOfIndividualsToInclude()){
    std::ostringstream os;
    os << "Number of individuals doesn't match in getNumberOfAllelesPerGenotype, they are "
    << snpVector.getNumberOfIndividualsToInclude() << " and " << phenotypeVector.getNumberOfIndividualsToInclude()
    << std::endl;
    const std::string& tmp = os.str();
    throw DimensionMismatch(tmp.c_str());
  }
#endif

  std::vector<int>* numberOfAllelesPerGenotype = new std::vector<int>(6);
  const int numberOfIndividuals = snpVector.getNumberOfIndividualsToInclude();
  const Container::RegularHostVector& snpData = snpVector.getOriginalSNPData();
  const Container::RegularHostVector& phenotypeData = phenotypeVector.getPhenotypeData();

  //UNROLL
  for(int i = 0; i < numberOfIndividuals; ++i){
    ++((*numberOfAllelesPerGenotype)[(int)(snpData(i) + 3 * phenotypeData(i))]);
  }

  return numberOfAllelesPerGenotype;
}

} /* namespace CPU */
} /* namespace CuEira */
