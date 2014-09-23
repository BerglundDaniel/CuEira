#ifndef DATAQUEUE_H_
#define DATAQUEUE_H_

#include <utility>
#include <vector>
#include <sstream>
#include <thread>
#include <mutex>

#include <Id.h>
#include <SNP.h>
#include <InvalidState.h>

#ifdef PROFILE
#include <boost/chrono/chrono_io.hpp>
#endif

namespace CuEira {
namespace Task {

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataQueue {
public:
  DataQueue(std::vector<SNP*>* snpQueue);
  virtual ~DataQueue();

  SNP* next();

  DataQueue(const DataQueue&) = delete;
  DataQueue(DataQueue&&) = delete;
  DataQueue& operator=(const DataQueue&) = delete;
  DataQueue& operator=(DataQueue&&) = delete;

private:
  std::vector<SNP*>* snpQueue;
  std::mutex mutex;

#ifdef PROFILE
  boost::chrono::duration<long long, boost::nano> timeWaitTotalLock;
#endif
};

} /* namespace Task */
} /* namespace CuEira */

#endif /* DATAQUEUE_H_ */
