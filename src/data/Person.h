#ifndef PERSON_H_
#define PERSON_H_

#include <string>

#include <Sex.h>
#include <Id.h>
#include <Phenotype.h>

namespace CuEira {

/**
 * This class contains the information about a person, its id, sex and associated row in the data.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class Person {
public:
  explicit Person(Id id, Sex sex, Phenotype phenotype);
  virtual ~Person();

  Id getId();
  Sex getSex();
  Phenotype getPhenotype();
  void setInclude(bool include);
  bool getInclude();

private:
  bool shouldPersonBeIncluded();

  Id id;
  Sex sex;
  Phenotype phenotype;
  bool include;
};

} /* namespace CuEira */

#endif /* PERSON_H_ */
