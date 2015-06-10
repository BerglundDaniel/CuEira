#include "Device.h"

namespace CuEira {
namespace CUDA {

Device::Device(int deviceNumber) :
    deviceNumber(deviceNumber){

}

Device::~Device(){

}

bool Device::isActive() const{
  int activeDeviceNumber;
  cudaGetDevice(&activeDeviceNumber);

  if(activeDeviceNumber == deviceNumber){
    return true;
  }else{
    return false;
  }

}

bool Device::setActiveDevice() const{
  cudaError_t status = cudaSetDevice(deviceNumber);

  if(status == cudaSuccess){
    return true;
  }else{
    return false;
  }
}

} /* namespace CUDA */
} /* namespace CuEira */
