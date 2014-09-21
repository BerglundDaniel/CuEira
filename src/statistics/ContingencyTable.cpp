#include "ContingencyTable.h"

namespace CuEira {

ContingencyTable::ContingencyTable(const std::vector<int>* tableCellNumbers) :
    tableCellNumbers(tableCellNumbers) {

}

ContingencyTable::~ContingencyTable() {
  delete tableCellNumbers;
}

const std::vector<int>& ContingencyTable::getTable() const {
  return *tableCellNumbers;
}

void ContingencyTable::toOstream(std::ostream& os) const {
  os << (*tableCellNumbers)[SNP0_ENV0_CASE_POSITION] << "," << (*tableCellNumbers)[SNP0_ENV0_CONTROL_POSITION] << ","
      << (*tableCellNumbers)[SNP1_ENV0_CASE_POSITION] << "," << (*tableCellNumbers)[SNP1_ENV0_CONTROL_POSITION] << ","
      << (*tableCellNumbers)[SNP0_ENV1_CASE_POSITION] << "," << (*tableCellNumbers)[SNP0_ENV1_CONTROL_POSITION] << ","
      << (*tableCellNumbers)[SNP1_ENV1_CASE_POSITION] << "," << (*tableCellNumbers)[SNP1_ENV1_CONTROL_POSITION];

}

std::ostream& operator<<(std::ostream& os, const ContingencyTable& contingencyTable) {
  contingencyTable.toOstream(os);
  return os;
}

} /* namespace CuEira */
