#include "SNPVectorFactory.h"

namespace CuEira {
namespace Container {

SNPVectorFactory::SNPVectorFactory(const Configuration& configuration) :
    configuration(configuration), geneticModel(configuration.getGeneticModel()) {

}

SNPVectorFactory::~SNPVectorFactory() {

}

} /* namespace Container */
} /* namespace CuEira */
