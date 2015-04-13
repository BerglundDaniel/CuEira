#include "CpuSNPVectorFactory.h"

namespace CuEira {
namespace Container {
namespace CPU {

CpuSNPVectorFactory::CpuSNPVectorFactory(const Configuration& configuration) :
    SNPVectorFactory(configuration) {

}

CpuSNPVectorFactory::~CpuSNPVectorFactory() {

}

SNPVector<RegularHostVector>* CpuSNPVectorFactory::constructSNPVector(SNP& snp, const HostVector* originalSNPData,
    const std::set<int>* snpMissingData) const {

  const int newSize = originalSNPData->getNumberOfRows() - snpMissingData->size();
  originalSNPData->updateSize(newSize);

  return new CpuSNPVector(snp, geneticModel, originalSNPData, snpMissingData);
}

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */
