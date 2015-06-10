#include "CpuSNPVectorFactory.h"

namespace CuEira {
namespace Container {
namespace CPU {

CpuSNPVectorFactory::CpuSNPVectorFactory(const Configuration& configuration) :
    SNPVectorFactory(configuration){

}

CpuSNPVectorFactory::~CpuSNPVectorFactory(){

}

CpuSNPVector* CpuSNPVectorFactory::constructSNPVector(SNP& snp, RegularHostVector* originalSNPData,
    const std::set<int>* snpMissingData) const{
  updateSize(originalSNPData, snpMissingData);
  return new CpuSNPVector(snp, geneticModel, originalSNPData, snpMissingData);
}

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */
