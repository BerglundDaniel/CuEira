#include "ModelInformationFactory.h"

namespace CuEira {
namespace Model {

ModelInformationFactory::ModelInformationFactory() {

}

ModelInformationFactory::~ModelInformationFactory() {

}

ModelInformation* ModelInformationFactory::constructModelInformation(ModelState modelState) const {
  return new ModelInformation(modelState, "");
}

ModelInformation* ModelInformationFactory::constructModelInformation(ModelState modelState, const SNP& snp,
    const EnvironmentFactor& environmentFactor, const AlleleStatistics& alleleStatistics) const {

  std::ostringstream informationStream;
  informationStream << snp << "," << environmentFactor << "," << alleleStatistics;

  return new ModelInformation(modelState, informationStream.str());
}

ModelInformation* ModelInformationFactory::constructModelInformation(ModelState modelState, const SNP& snp,
    const EnvironmentFactor& environmentFactor, const AlleleStatistics& alleleStatistics,
    const ContingencyTable& contingencyTable) const {

  std::ostringstream informationStream;
  informationStream << snp << "," << environmentFactor << "," << alleleStatistics << "," << contingencyTable;

  return new ModelInformation(modelState, informationStream.str());
}

} /* namespace Model */
} /* namespace CuEira */
