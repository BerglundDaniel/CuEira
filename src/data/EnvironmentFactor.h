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
  EnvironmentFactor(Id id);
  virtual ~EnvironmentFactor();

  Id getId() const;
  bool getInclude() const;

private:
  bool shouldEnvironmentFactorBeIncluded() const;

  Id id;
  bool include;
};

} /* namespace CuEira */

#endif /* ENVIRONMENTFACTOR_H_ */
