#include "DataQueue.h"

namespace CuEira {
namespace Task {

DataQueue::DataQueue(std::vector<SNP*> snpQueue) :
    snpQueue(snpQueue) {

}

DataQueue::~DataQueue() {

}

bool DataQueue::hasNext() {
  return !snpQueue.empty();
}

SNP* DataQueue::next() {
#ifdef DEBUG
  if(snpQueue.empty()){
    throw new InvalidState("Vector of SNPs is empty in DataQueue.");
  }
#endif
  SNP* currentSNP = snpQueue.back();
  snpQueue.pop_back();

  return currentSNP;
}

} /* namespace Task */
} /* namespace CuEira */
