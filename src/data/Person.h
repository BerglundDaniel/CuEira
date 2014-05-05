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

  Id getId() const;
  Sex getSex() const;
  Phenotype getPhenotype() const;
  bool getInclude() const;

  bool operator<(const Person& otherPerson) const;
  bool operator==(const Person& otherPerson) const;

private:
  bool shouldPersonBeIncluded() const;

  Id id;
  Sex sex;
  Phenotype phenotype;
  bool include;
};

} /* namespace CuEira */

#endif /* PERSON_H_ */
