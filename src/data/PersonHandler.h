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
#include <HostVector.h>
#include <InvalidState.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
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
  PersonHandler();
  virtual ~PersonHandler();

  virtual const Person& createPerson(Id id, Sex sex, Phenotype phenotype, int rowAll);

  virtual int getNumberOfIndividualsTotal() const;
  virtual int getNumberOfIndividualsToInclude() const;
  virtual const Person& getPersonFromId(Id id) const;
  virtual const Person& getPersonFromRowAll(int row) const;
  virtual const Person& getPersonFromRowInclude(int row) const;
  virtual int getRowIncludeFromPerson(const Person& person) const;
  virtual const Container::HostVector& getOutcomes() const;
  virtual void createOutcomes();

  PersonHandler(const PersonHandler&) = delete;
  PersonHandler(PersonHandler&&) = delete;
  PersonHandler& operator=(const PersonHandler&) = delete;
  PersonHandler& operator=(PersonHandler&&) = delete;

private:
  bool shouldPersonBeIncluded(Id id, Sex sex, Phenotype phenotype) const;

  int numberOfIndividualsTotal;
  int numberOfIndividualsToInclude;
  std::map<int, Person*> rowToPersonAll;
  std::map<int, Person*> rowToPersonInclude;
  std::map<Id, Person*> idToPerson;
  std::map<const Person*, int> personToRowInclude;
  bool outcomesCreated;
  Container::HostVector* outcomes;
};

} /* namespace CuEira */

#endif /* PERSONHANDLER_H_ */
