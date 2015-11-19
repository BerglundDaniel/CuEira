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
#include <PersonHandlerException.h>

namespace CuEira {
class PersonHandlerLocked;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PersonHandler {
  friend PersonHandlerLocked;
public:
  typedef std::vector<Person*>::const_iterator iterator;

  explicit PersonHandler(std::vector<Person*>* persons);
  virtual ~PersonHandler();

  virtual int getNumberOfIndividualsTotal() const;

  virtual iterator begin() noexcept;
  virtual iterator end() noexcept;

  virtual const Person& getPersonFromId(Id id) const;
  virtual Person& getPersonFromId(Id id);
  virtual const Person& getPersonFromRowAll(int rowAll) const;
  virtual Person& getPersonFromRowAll(int rowAll);

  PersonHandler(const PersonHandler&) = delete;
  PersonHandler(PersonHandler&&) = delete;
  PersonHandler& operator=(const PersonHandler&) = delete;
  PersonHandler& operator=(PersonHandler&&) = delete;

private:
  const int numberOfIndividualsTotal;
  std::vector<Person*>* persons;
  std::map<Id, Person*> idToPerson;
};

} /* namespace CuEira */

#endif /* PERSONHANDLER_H_ */
