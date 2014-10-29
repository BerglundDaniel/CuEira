#ifndef DEVICETOHOSTMOCK_H_
#define DEVICETOHOSTMOCK_H_

#include <gmock/gmock.h>

#include <DeviceToHost.h>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <PinnedHostVector.h>
#include <PinnedHostMatrix.h>

namespace CuEira {
namespace CUDA {

class DeviceToHostMock: public DeviceToHost {
public:
  DeviceToHostMock() :
      DeviceToHost() {

  }

  virtual ~DeviceToHostMock() {

  }

  MOCK_CONST_METHOD1(transferMatrix, Container::PinnedHostMatrix*(const Container::DeviceMatrix&));
  MOCK_CONST_METHOD1(transferVector, Container::PinnedHostVector*(const Container::DeviceVector&));

  MOCK_CONST_METHOD2(transferMatrix, void(const Container::DeviceMatrix&, PRECISION*));
  MOCK_CONST_METHOD2(transferVector, void(const Container::DeviceVector&, PRECISION*));

};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* DEVICETOHOSTMOCK_H_ */
