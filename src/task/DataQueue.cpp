#include "DataQueue.h"

namespace CuEira {
namespace Task {

DataQueue::DataQueue() {

}

DataQueue::~DataQueue() {
  //FIXME should contents be deleted? who resposnible

  delete snpQueue;
  delete environmentQueue;
}

bool DataQueue::hasNext() {

}

std::pair<SNP*, EnvironmentFactor*>* DataQueue::next() {


  return new std::pair<SNP*, EnvironmentFactor*>(currentSNP, currentEnvironmentFactor);
}

} /* namespace Task */
} /* namespace CuEira */
