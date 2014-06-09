#ifndef ENVIRONMENTFACTOR_H_
#define ENVIRONMENTFACTOR_H_

#include <Id.h>
#include <VariableType.h>

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
  void setVariableType(VariableType variableType);
  VariableType getVariableType() const;

  bool operator<(const EnvironmentFactor& otherEnvironmentFactor) const;
  bool operator==(const EnvironmentFactor& otherEnvironmentFactor) const;

private:
  bool shouldEnvironmentFactorBeIncluded() const;

  Id id;
  bool include;
  VariableType variableType;
};

} /* namespace CuEira */

#endif /* ENVIRONMENTFACTOR_H_ */
