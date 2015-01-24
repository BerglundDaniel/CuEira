#ifndef EVENT_H_
#define EVENT_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <Stream.h>
#include <CudaAdapter.cu>
#include <CudaException.h>

namespace CuEira {
namespace CUDA {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class Event {
public:
  Event(const Stream& stream);
  virtual ~Event();

  float operator-(Event& otherEvent);

#ifndef __CUDACC__
  Event(const Event&) = delete;
  Event(Event&&) = delete;
  Event& operator=(const Event&) = delete;
  Event& operator=(Event&&) = delete;
#endif

  cudaEvent_t cudaEvent;

private:
  const Stream& stream;
};

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* EVENT_H_ */
