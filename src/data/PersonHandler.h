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

  void addPerson(Person person, int rowAll);

  int getNumberOfIndividualsTotal() const;
  int getNumberOfIndividualsToInclude() const;
  const Person& getPersonFromId(Id id) const;
  const Person& getPersonFromRowAll(int row) const;
  const Person& getPersonFromRowInclude(int row) const;
  int getRowIncludeFromPerson(Person& person) const;

private:
  int numberOfIndividualsTotal;
  int numberOfIndividualsToInclude;
  std::map<int, Person> rowToPersonAll;
  std::map<int, Person> rowToPersonInclude;
  std::map<Id, Person> idToPerson;
  std::map<Person, int> personToRowInclude;
};

} /* namespace CuEira */

#endif /* PERSONHANDLER_H_ */
