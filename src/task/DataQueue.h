#ifndef DATAQUEUE_H_
#define DATAQUEUE_H_

#include <utility>
#include <vector>

#include <Id.h>
#include <SNP.h>
#include <EnvironmentFactor.h>

namespace CuEira {
namespace Task {

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataQueue {
public:
  DataQueue();
  virtual ~DataQueue();

  bool hasNext();
  std::pair<SNP*, EnvironmentFactor*>* next();

private:
  std::vector<SNP*>* snpQueue;
  std::vector<EnvironmentFactor*>* environmentQueue;

  SNP* currentSNP;
  EnvironmentFactor* currentEnvironmentFactor;
};

} /* namespace Task */
} /* namespace CuEira */

#endif /* DATAQUEUE_H_ */
