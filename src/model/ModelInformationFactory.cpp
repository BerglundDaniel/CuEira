#include "ModelInformationFactory.h"

namespace CuEira {
namespace Model {

ModelInformationFactory::ModelInformationFactory() {

}

ModelInformationFactory::~ModelInformationFactory() {

}

ModelInformation* ModelInformationFactory::constructModelInformation(const SNP& snp,
    const EnvironmentFactor& environmentFactor, const AlleleStatistics& alleleStatistics) const {
  return new ModelInformation(snp, environmentFactor, alleleStatistics);
}

ModelInformation* ModelInformationFactory::constructModelInformation(const SNP& snp,
    const EnvironmentFactor& environmentFactor, const AlleleStatistics& alleleStatistics,
    const ContingencyTable& contingencyTable) const {
  return new ModelInformationWithTable(snp, environmentFactor, alleleStatistics, contingencyTable);
}

} /* namespace Model */
} /* namespace CuEira */
