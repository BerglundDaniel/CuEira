#ifndef PERSONHANDLER_H_
#define PERSONHANDLER_H_

#include <sstream>
#include <string>
#include <map>
#include <vector>

#include <Id.h>
#include <Sex.h>
#include <Person.h>
#include <Phenotype.h>
#include <Configuration.h>
#include <PersonHandlerException.h>
#include <HostVector.h>
#include <InvalidState.h>

#ifdef CPU
#include <RegularHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PersonHandler {
public:
  explicit PersonHandler(std::vector<Person*>* persons);
  virtual ~PersonHandler();

  virtual int getNumberOfIndividualsTotal() const;
  virtual int getNumberOfIndividualsToInclude() const;
  virtual const std::vector<Person*>& getPersons() const;

  virtual const Person& getPersonFromId(Id id) const;
  virtual Person& getPersonFromId(Id id);
  virtual const Person& getPersonFromRowAll(int rowAll) const;
  virtual int getRowIncludeFromPerson(const Person& person) const;

  virtual void lockIndividuals();

  PersonHandler(const PersonHandler&) = delete;
  PersonHandler(PersonHandler&&) = delete;
  PersonHandler& operator=(const PersonHandler&) = delete;
  PersonHandler& operator=(PersonHandler&&) = delete;

private:
  const int numberOfIndividualsTotal;
  int numberOfIndividualsToInclude;
  std::vector<Person*>* persons;
  std::map<Id, Person*> idToPerson;
  std::map<Person*, int> personToRowInclude;
  bool individualsLocked;
};

} /* namespace CuEira */

#endif /* PERSONHANDLER_H_ */
