#ifndef CONTINGENCYTABLEMOCK_H_
#define CONTINGENCYTABLEMOCK_H_

#include <vector>
#include <gmock/gmock.h>

#include <ContingencyTable.h>

namespace CuEira {

class ContingencyTableMock: public ContingencyTable {
public:
  ContingencyTableMock() :
  ContingencyTable(nullptr){

  }

  virtual ~ContingencyTableMock() {

  }

  MOCK_CONST_METHOD0(getTable, const std::vector<int>&());
};

} /* namespace CuEira */

#endif /* CONTINGENCYTABLEMOCK_H_ */
