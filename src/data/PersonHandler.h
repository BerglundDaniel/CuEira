#ifndef PERSONHANDLER_H_
#define PERSONHANDLER_H_

#include <sstream>
#include <string>
#include <map>

#include <Id.h>
#include <Sex.h>
#include <Person.h>
#include <Phenotype.h>
#include <Configuration.h>
#include <PersonHandlerException.h>

namespace CuEira {
/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PersonHandler {
public:
  PersonHandler();
  virtual ~PersonHandler();

  virtual const Person& createPerson(Id id, Sex sex, Phenotype phenotype, int rowAll);

  virtual int getNumberOfIndividualsTotal() const;
  virtual int getNumberOfIndividualsToInclude() const;
  virtual const Person& getPersonFromId(Id id) const;
  virtual const Person& getPersonFromRowAll(int row) const;
  virtual const Person& getPersonFromRowInclude(int row) const;
  virtual int getRowIncludeFromPerson(const Person& person) const;

private:
  bool shouldPersonBeIncluded(Id id, Sex sex, Phenotype phenotype) const;

  int numberOfIndividualsTotal;
  int numberOfIndividualsToInclude;
  std::map<int, Person> rowToPersonAll;
  std::map<int, Person> rowToPersonInclude;
  std::map<Id, Person> idToPerson;
  std::map<Person, int> personToRowInclude;
};

} /* namespace CuEira */

#endif /* PERSONHANDLER_H_ */
