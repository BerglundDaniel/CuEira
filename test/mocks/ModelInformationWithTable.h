#ifndef MODELINFORMATIONWITHTABLEMOCK_H_
#define MODELINFORMATIONWITHTABLEMOCK_H_

#include <gmock/gmock.h>
#include <string>
#include <ostream>

#include <ModelInformationWithTable.h>

namespace CuEira {
namespace Model {

class ModelInformationWithTableMock: public ModelInformationWithTable {
public:
  ModelInformationWithTableMock() :
      ModelInformationWithTable() {

  }

  virtual ~ModelInformationWithTableMock() {

  }

  MOCK_CONST_METHOD1(toOstream, void(std::ostream& os));
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* MODELINFORMATIONWITHTABLEMOCK_H_ */
