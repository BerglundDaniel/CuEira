#include "ModelInformation.h"

namespace CuEira {
namespace Model {

ModelInformation::ModelInformation(const SNP& snp, const EnvironmentFactor& environmentFactor,
    const AlleleStatistics& alleleStatistics) :
    snp(&snp), environmentFactor(&environmentFactor), alleleStatistics(&alleleStatistics) {

}

ModelInformation::~ModelInformation() {

}

ModelInformation::ModelInformation() :
    snp(nullptr), environmentFactor(nullptr), alleleStatistics(nullptr) {

}

void ModelInformation::toOstream(std::ostream& os) const {
  os << *snp << "," << *environmentFactor << "," << *alleleStatistics;
}

std::ostream & operator<<(std::ostream& os, const ModelInformation& modelInformation) {
  modelInformation.toOstream(os);
  return os;
}

} /* namespace Model */
} /* namespace CuEira */
