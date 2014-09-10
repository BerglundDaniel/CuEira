#ifndef STREAMFACTORYMOCK_H_
#define STREAMFACTORYMOCK_H_

#include <gmock/gmock.h>

#include <Stream.h>
#include <StreamFactory.h>
#include <Device.h>

namespace CuEira {
namespace CUDA {

class StreamFactoryMock: public StreamFactory {
public:
  StreamFactoryMock() :
      StreamFactory() {

  }

  virtual ~StreamFactoryMock() {

  }

  MOCK_CONST_METHOD1(constructStream, Stream*(const Device& device));
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* STREAMFACTORYMOCK_H_ */
