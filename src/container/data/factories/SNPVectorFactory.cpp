#include "SNPVectorFactory.h"

namespace CuEira {
namespace Container {

template<typename Vector, typename VectorSNP>
SNPVectorFactory<Vector, VectorSNP>::SNPVectorFactory(const Configuration& configuration) :
    configuration(configuration), geneticModel(configuration.getGeneticModel()){

}

template<typename Vector, typename VectorSNP>
SNPVectorFactory<Vector, VectorSNP>::~SNPVectorFactory(){

}

template<typename Vector, typename VectorSNP>
void SNPVectorFactory<Vector, VectorSNP>::updateSize(Vector* originalSNPData,
    const std::set<int>* snpMissingData) const{
  const int newSize = originalSNPData->getNumberOfRows() - snpMissingData->size();
  originalSNPData->updateSize(newSize);
}

} /* namespace Container */
} /* namespace CuEira */
