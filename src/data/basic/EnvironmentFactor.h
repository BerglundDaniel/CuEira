#ifndef ENVIRONMENTFACTOR_H_
#define ENVIRONMENTFACTOR_H_

#include <ostream>

#include <Id.h>
#include <VariableType.h>

namespace CuEira {

/**
 * This class contains information about a column of environment factors
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class EnvironmentFactor {
  friend std::ostream& operator<<(std::ostream& os, const EnvironmentFactor& envFactor);
public:
  EnvironmentFactor(Id id);
  virtual ~EnvironmentFactor();

  Id getId() const;
  void setVariableType(VariableType variableType);
  VariableType getVariableType() const;

  void setMax(int max);
  void setMin(int min);
  int getMax() const;
  int getMin() const;

  bool operator<(const EnvironmentFactor& otherEnvironmentFactor) const;
  bool operator==(const EnvironmentFactor& otherEnvironmentFactor) const;

  EnvironmentFactor(const EnvironmentFactor&) = delete;
  EnvironmentFactor(EnvironmentFactor&&) = delete;
  EnvironmentFactor& operator=(const EnvironmentFactor&) = delete;
  EnvironmentFactor& operator=(EnvironmentFactor&&) = delete;

private:
  Id id;
  VariableType variableType;
  int max;
  int min;
};

} /* namespace CuEira */

#endif /* ENVIRONMENTFACTOR_H_ */
