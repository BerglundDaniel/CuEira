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
    Die();
  }

  MOCK_CONST_METHOD0(getTable, const std::vector<int>&());
  MOCK_METHOD0(Die, void());

};

} /* namespace CuEira */

#endif /* CONTINGENCYTABLEMOCK_H_ */
