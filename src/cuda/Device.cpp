#include "Device.h"

namespace CuEira {
namespace CUDA {

Device::Device(int deviceNumber) :
    deviceNumber(deviceNumber), outcomes(nullptr), outcomesSet(false) {

}

Device::~Device() {
  delete outcomes;
}

bool Device::isActive() const {
  int activeDeviceNumber;
  cudaGetDevice(&activeDeviceNumber);

  if(activeDeviceNumber == deviceNumber){
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

void Device::setOutcomes(const Container::DeviceVector* outcomes) {
  outcomesSet = true;
  this->outcomes = outcomes;
}

const Container::DeviceVector& Device::getOutcomes() const {
  if(!outcomesSet){
    throw new InvalidState("Outcomes not set for Device.");
  }
  return *outcomes;
}

} /* namespace CUDA */
} /* namespace CuEira */
