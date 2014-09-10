#include "Device.h"

namespace CuEira {
namespace CUDA {

Device::Device(int deviceNumber) :
    deviceNumber(deviceNumber), outcomes(nullptr) {

}

Device::~Device() {
  delete outcomes;
}

bool Device::isActive() const {
  int* activeDeviceNumber = new int(-1);
  cudaGetDevice(activeDeviceNumber);

  if(*activeDeviceNumber == deviceNumber){
    return true;
  }else{
    return false;
  }
}

bool Device::setActiveDevice() const {
  cudaError_t status = cudaSetDevice(deviceNumber);

  if(status == cudaSuccess){
    return true;
  }else{
    return false;
  }
}

const void Device::setOutcomes(const Container::DeviceVector* outcomes) {
  this->outcomes = outcomes;
}

const Container::DeviceVector& Device::getOutcomes() const {
  return *outcomes;
}

} /* namespace CUDA */
} /* namespace CuEira */
