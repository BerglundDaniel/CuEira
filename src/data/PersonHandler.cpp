#include "PersonHandler.h"

namespace CuEira {

PersonHandler::PersonHandler() :
    numberOfIndividualsTotal(0), numberOfIndividualsToInclude(0) {

}

PersonHandler::~PersonHandler() {

}

const Person& PersonHandler::createPerson(Id id, Sex sex, Phenotype phenotype, int rowAll) {
  if(rowToPersonAll.count(rowAll) > 0){
    std::ostringstream os;
    os << "There already is a person with plink row: " << rowAll << " id: " << id.getString() << std::endl;
    const std::string& tmp = os.str();
    throw PersonHandlerException(tmp.c_str());
  }
  if(idToPerson.count(id) > 0){
    std::ostringstream os;
    os << "There already is a person with id: " << id.getString() << std::endl;
    const std::string& tmp = os.str();
    throw PersonHandlerException(tmp.c_str());
  }

  bool include = shouldPersonBeIncluded(id, sex, phenotype);
  Person person(id, sex, phenotype, include);

  idToPerson.insert(std::pair<Id, Person>(person.getId(), person));
  rowToPersonAll.insert(std::pair<int, Person>(rowAll, person));

  if(include){
    rowToPersonInclude.insert(std::pair<int, Person>(numberOfIndividualsToInclude, person));
    personToRowInclude.insert(std::pair<Person, int>(person, numberOfIndividualsToInclude));

    numberOfIndividualsToInclude++;
  }
  numberOfIndividualsTotal++;

  return idToPerson.at(id);
}

int PersonHandler::getNumberOfIndividualsTotal() const {
  return numberOfIndividualsTotal;
}

int PersonHandler::getNumberOfIndividualsToInclude() const {
  return numberOfIndividualsToInclude;
}

const Person& PersonHandler::getPersonFromId(Id id) const {
  if(idToPerson.count(id) <= 0){
    std::ostringstream os;
    os << "No person with id " << id.getString() << std::endl;
    const std::string& tmp = os.str();
    throw PersonHandlerException(tmp.c_str());
  }
  return idToPerson.at(id);
}

const Person& PersonHandler::getPersonFromRowAll(int row) const {
  if(rowToPersonAll.count(row) <= 0){
    std::ostringstream os;
    os << "No person from row all: " << row << std::endl;
    const std::string& tmp = os.str();
    throw PersonHandlerException(tmp.c_str());
  }
  return rowToPersonAll.at(row);
}

const Person& PersonHandler::getPersonFromRowInclude(int row) const {
  if(rowToPersonInclude.count(row) <= 0){
    std::ostringstream os;
    os << "No person from row include: " << row << std::endl;
    const std::string& tmp = os.str();
    throw PersonHandlerException(tmp.c_str());
  }
  return rowToPersonInclude.at(row);
}

int PersonHandler::getRowIncludeFromPerson(const Person& person) const {
  if(personToRowInclude.count(person) <= 0){
    std::ostringstream os;
    os << "Person not included: " << person.getId().getString() << std::endl;
    const std::string& tmp = os.str();
    throw PersonHandlerException(tmp.c_str());
  }
  return personToRowInclude.at(person);
}

bool PersonHandler::shouldPersonBeIncluded(Id id, Sex sex, Phenotype phenotype) const {
  if(phenotype == MISSING){
    return false;
  }else{
    return true;
  }
}

} /* namespace CuEira */
