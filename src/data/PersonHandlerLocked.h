#ifndef PERSONHANDLERLOCKED_H_
#define PERSONHANDLERLOCKED_H_

#include <sstream>
#include <string>
#include <map>
#include <vector>

#include <Id.h>
#include <Sex.h>
#include <Person.h>
#include <Phenotype.h>
#include <PersonHandler.h>
#include <PersonHandlerException.h>

namespace CuEira {

/*
 * This class....
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PersonHandlerLocked {
public:
  typedef std::vector<Person*>::const_iterator const_iterator;

  explicit PersonHandlerLocked(PersonHandler& personHandler);
  virtual ~PersonHandlerLocked();

  virtual int getNumberOfIndividualsTotal() const;
  virtual int getNumberOfIndividualsToInclude() const;

  virtual const_iterator begin() const noexcept;
  virtual const_iterator end() const noexcept;

  virtual const Person& getPersonFromId(Id id) const;
  virtual const Person& getPersonFromRowAll(int rowAll) const;
  virtual int getRowIncludeFromPerson(const Person& person) const;

  PersonHandlerLocked(const PersonHandlerLocked&) = delete;
  PersonHandlerLocked(PersonHandlerLocked&&) = delete;
  PersonHandlerLocked& operator=(const PersonHandlerLocked&) = delete;
  PersonHandlerLocked& operator=(PersonHandlerLocked&&) = delete;

private:
  template<class T> struct pointerLess {
    bool operator()(T* left, T* right) const{
      return *left < *right;
    }
  };

  const int numberOfIndividualsTotal;
  int numberOfIndividualsToInclude;
  const std::vector<Person*>* persons;
  std::map<Id, Person*>* idToPerson;
  std::map<const Person*, int, pointerLess<const Person> > personToRowInclude;
};

} /* namespace CuEira */

#endif /* PERSONHANDLERLOCKED_H_ */
