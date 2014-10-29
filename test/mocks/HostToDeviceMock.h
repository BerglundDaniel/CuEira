#ifndef HOSTTODEVICEMOCK_H_
#define HOSTTODEVICEMOCK_H_

#include <gmock/gmock.h>

#include <HostToDevice.h>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <PinnedHostVector.h>
#include <PinnedHostMatrix.h>

namespace CuEira {
namespace CUDA {

class HostToDeviceMock: public HostToDevice {
public:
  HostToDeviceMock() :
      HostToDevice() {

  }

  virtual ~HostToDeviceMock() {

  }

  MOCK_CONST_METHOD1(transferMatrix, Container::DeviceMatrix*(const Container::PinnedHostMatrix&));
  MOCK_CONST_METHOD1(transferVector, Container::DeviceVector*(const Container::PinnedHostVector&));

  MOCK_CONST_METHOD2(transferMatrix, void(const Container::PinnedHostMatrix&, PRECISION*));
  MOCK_CONST_METHOD2(transferVector, void(const Container::PinnedHostVector&, PRECISION*));

};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* HOSTTODEVICEMOCK_H_ */
