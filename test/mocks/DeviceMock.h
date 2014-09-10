#ifndef DEVICEMOCK_H_
#define DEVICEMOCK_H_

#include <gmock/gmock.h>

#include <Device.h>
#include <DeviceVector.h>

namespace CuEira {
namespace CUDA {

class DeviceMock: public Device {
public:
  DeviceMock() :
  Device(0, nullptr){

  }

  virtual ~DeviceMock() {

  }

  MOCK_CONST_METHOD0(isActive, bool());
  MOCK_CONST_METHOD0(setActiveDevice, bool());
  MOCK_CONST_METHOD0(getOutcomes, const Container::DeviceVector&());
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* DEVICEMOCK_H_ */
