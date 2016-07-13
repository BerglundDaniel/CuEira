#include "SNPVectorFactory.h"

namespace CuEira {
namespace Container {

template<typename Vector>
SNPVectorFactory<Vector>::SNPVectorFactory(const Configuration& configuration) :
    configuration(configuration), geneticModel(configuration.getGeneticModel()){

}

template<typename Vector>
SNPVectorFactory<Vector>::~SNPVectorFactory(){

}
/*
template<typename Vector, typename U = Vector, typename std::enable_if<!(typeid(U) == typeid(DeviceVector)),int>::type =
    0>
SNPVector<U>* SNPVectorFactory<Vector>::constructSNPVector(SNP& snp, U* originalSNPData,
    const std::set<int>* snpMissingData) const{

}

template<typename Vector, typename U = Vector, typename std::enable_if<(typeid(U) == typeid(DeviceVector)),int>::type =
    0>
SNPVector<U>* SNPVectorFactory<Vector>::constructSNPVector(SNP& snp, PinnedHostVector* originalSNPData,
    const std::set<int>* snpMissingData) const{

}*/

template<typename Vector>
void SNPVectorFactory<Vector>::updateSize(Vector* originalSNPData, const std::set<int>* snpMissingData) const{
  const int newSize = originalSNPData->getNumberOfRows() - snpMissingData->size();
  originalSNPData->updateSize(newSize);
}

} /* namespace Container */
} /* namespace CuEira */
