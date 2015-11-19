#include "CudaSNPVectorFactory.h"

namespace CuEira {
namespace Container {
namespace CUDA {

CudaSNPVectorFactory::CudaSNPVectorFactory(const Configuration& configuration, const Stream& stream) :
    SNPVectorFactory(configuration), stream(stream){

}

CudaSNPVectorFactory::~CudaSNPVectorFactory(){

}

CudaSNPVector* CudaSNPVectorFactory::constructSNPVector(SNP& snp, PinnedHostVector* originalSNPData,
    const std::set<int>* snpMissingData) const{

  updateSize(originalSNPData, snpMissingData);
  DeviceVector* originalSNPDataDevice = transferVector(stream, *originalSNPData);
  delete originalSNPData;

  return new CudaSNPVector(snp, geneticModel, originalSNPDataDevice, snpMissingData, stream);
}

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */
