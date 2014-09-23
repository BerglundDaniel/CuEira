#include "DataQueue.h"

namespace CuEira {
namespace Task {

DataQueue::DataQueue(std::vector<SNP*>* snpQueue) :
    snpQueue(snpQueue) {

}

DataQueue::~DataQueue() {
  delete snpQueue;
#ifdef PROFILE
  std::cerr << "DataQueue, time spent waiting at locks: " << boost::chrono::duration_cast<boost::chrono::microseconds>(timeWaitTotalLock) << std::endl;
#endif
}

SNP* DataQueue::next() {
  if(snpQueue->empty()){
    return nullptr;
  }
#ifdef PROFILE
  boost::chrono::system_clock::time_point beforeLock = boost::chrono::system_clock::now();
#endif
  mutex.lock();
#ifdef PROFILE
  boost::chrono::system_clock::time_point afterLock = boost::chrono::system_clock::now();
  timeWaitTotalLock+=afterLock - beforeLock;
#endif

  if(snpQueue->empty()){
    mutex.unlock();
    return nullptr;
  }

  SNP* currentSNP = snpQueue->back();
  snpQueue->pop_back();

  mutex.unlock();
  return currentSNP;
}

} /* namespace Task */
} /* namespace CuEira */
