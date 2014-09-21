#ifndef MODELINFORMATIONMOCK_H_
#define MODELINFORMATIONMOCK_H_

#include <gmock/gmock.h>
#include <string>
#include <ostream>

#include <ModelInformation.h>

namespace CuEira {
namespace Model {

class ModelInformationMock: public ModelInformation {
public:
  ModelInformationMock() :
      ModelInformation(DONE, "") {

  }

  virtual ~ModelInformationMock() {

  }

  MOCK_CONST_METHOD0(getModelState, ModelState());

  MOCK_CONST_METHOD1(toOstream, void(std::ostream& os));
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* MODELINFORMATIONMOCK_H_ */
