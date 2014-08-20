#include "DataQueue.h"

namespace CuEira {
namespace Task {

DataQueue::DataQueue(std::vector<SNP*> snpQueue) :
    snpQueue(snpQueue) {

}

DataQueue::~DataQueue() {

}

SNP* DataQueue::next() {
  mutex.lock();

  if(snpQueue.empty()){
    return nullptr;
  }

  SNP* currentSNP = snpQueue.back();
  snpQueue.pop_back();

  mutex.unlock();
  return currentSNP;
}

} /* namespace Task */
} /* namespace CuEira */
