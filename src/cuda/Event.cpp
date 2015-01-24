#include "Event.h"

namespace CuEira {
namespace CUDA {

Event::Event(const Stream& stream) :
    stream(stream) {
  handleCudaStatus(cudaEventCreate(&cudaEvent, cudaEventDefault), "Failed to create CUDA event");
  handleCudaStatus(cudaEventRecord(cudaEvent, stream.getCudaStream()), "Failed to record CUDA event on stream.");
}

Event::~Event() {

}

float Event::operator-(Event& otherEvent) {
  float timeElapsed;
  handleCudaStatus(cudaEventElapsedTime(&timeElapsed, otherEvent.cudaEvent, cudaEvent),
      "Failed to get elapsed time between CUDA events.");
  return timeElapsed;
}

} /* namespace CUDA */
} /* namespace CuEira */
