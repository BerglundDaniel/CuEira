#include "SNPVectorFactory.h"

namespace CuEira {
namespace Container {

SNPVectorFactory::SNPVectorFactory(const Configuration& configuration) :
    configuration(configuration), geneticModel(configuration.getGeneticModel()) {

}

SNPVectorFactory::~SNPVectorFactory() {

}

SNPVector* SNPVectorFactory::constructSNPVector(SNP& snp, const HostVector* originalSNPData,
    const std::set<int>* snpMissingData) const {

  //TODO move to gpu and stuff

  return new SNPVector<>(snp, geneticModel, originalSNPData, snpMissingData);
}

} /* namespace Container */
} /* namespace CuEira */
