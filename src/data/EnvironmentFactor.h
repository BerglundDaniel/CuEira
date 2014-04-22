#ifndef ENVIRONMENTFACTOR_H_
#define ENVIRONMENTFACTOR_H_

#include <Id.h>

namespace CuEira {

/**
 * This class contains information about a column of environment factors
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class EnvironmentFactor {
public:
  EnvironmentFactor(Id id, bool include=true);
  virtual ~EnvironmentFactor();

  Id getId();
  bool getInclude();
  void setInclude(bool include);

private:
  Id id;
  bool include;
};

} /* namespace CuEira */

#endif /* ENVIRONMENTFACTOR_H_ */
