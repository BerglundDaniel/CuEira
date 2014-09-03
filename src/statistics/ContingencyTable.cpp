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

} /* namespace CuEira */
