#ifndef PERSON_H_
#define PERSON_H_

#include <string>
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <Sex.h>
#include <Id.h>
#include <Phenotype.h>

namespace CuEira {
class PersonTest;
class PersonHandler;
namespace CuEira_Test {
class ConstructorHelpers;
}

/**
 * This class contains the information about a person, its id, sex and associated row in the data.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class Person {
  friend PersonTest;
  FRIEND_TEST(PersonTest, Getters);
  FRIEND_TEST(PersonTest, Operators);
  friend PersonHandler;
  friend CuEira_Test::ConstructorHelpers;
public:
  virtual ~Person();

  Id getId() const;
  Sex getSex() const;
  Phenotype getPhenotype() const;
  bool getInclude() const;

  bool operator<(const Person& otherPerson) const;
  bool operator==(const Person& otherPerson) const;

  Person(const Person&) = delete;
  Person(Person&&) = delete;
  Person& operator=(const Person&) = delete;
  Person& operator=(Person&&) = delete;

protected:
  explicit Person(Id id, Sex sex, Phenotype phenotype, bool include);

private:
  const Id id;
  const Sex sex;
  const Phenotype phenotype;
  const bool include;
};

} /* namespace CuEira */

#endif /* PERSON_H_ */
