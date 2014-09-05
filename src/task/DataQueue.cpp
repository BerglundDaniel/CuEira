#include "DataQueue.h"

namespace CuEira {
namespace Task {

DataQueue::DataQueue(std::vector<SNP*>* snpQueue) :
    snpQueue(snpQueue) {

}

DataQueue::~DataQueue() {
  delete snpQueue;
}

SNP* DataQueue::next() {
  if(snpQueue->empty()){
    return nullptr;
  }

  mutex.lock();

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
