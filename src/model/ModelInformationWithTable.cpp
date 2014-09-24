#include "ModelInformationWithTable.h"

namespace CuEira {
namespace Model {

ModelInformationWithTable::ModelInformationWithTable(const SNP& snp, const EnvironmentFactor& environmentFactor,
    const AlleleStatistics& alleleStatistics, const ContingencyTable& contingencyTable) :
    ModelInformation(snp, environmentFactor, alleleStatistics), contingencyTable(&contingencyTable) {

}

ModelInformationWithTable::~ModelInformationWithTable() {

}

ModelInformationWithTable::ModelInformationWithTable() :
    ModelInformation(), contingencyTable(nullptr) {

}

void ModelInformationWithTable::toOstream(std::ostream& os) const {
  os << *snp << "," << *environmentFactor << "," << *alleleStatistics << "," << *contingencyTable;
}

std::ostream & operator<<(std::ostream& os, const ModelInformationWithTable& modelInformationWithTable) {
  modelInformationWithTable.toOstream(os);
  return os;
}

} /* namespace Model */
} /* namespace CuEira */
