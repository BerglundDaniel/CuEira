#include "PersonHandler.h"

namespace CuEira {

PersonHandler::PersonHandler() :
    numberOfIndividualsTotal(0), numberOfIndividualsToInclude(0) {

}

PersonHandler::~PersonHandler() {

}

void PersonHandler::addPerson(Person person, int individualNumber) {
  idToPerson[person.getId()] = person;
  rowToPersonAll[individualNumber] = person;

  if(person.getInclude()){
    rowToPersonInclude[numberOfIndividualsToInclude] = person;

    numberOfIndividualsToInclude++;
  }
  numberOfIndividualsTotal++;
}

int PersonHandler::getNumberOfIndividualsTotal() const {
  return numberOfIndividualsTotal;
}

int PersonHandler::getNumberOfIndividualsToInclude() const {
  return numberOfIndividualsToInclude;
}

Person& PersonHandler::getPersonFromId(Id id) const {
  if(idToPerson.count(id) <= 0){
    std::ostringstream os;
    os << "No person with id " << id.getString() << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
  return idToPerson[id];
}

Person& PersonHandler::getPersonFromRowAll(int row) const {
  if(rowToPersonAll.count(row) <= 0){
    std::ostringstream os;
    os << "No person from row all: " << row << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
  return rowToPersonAll[row];
}

Person& PersonHandler::getPersonFromRowInclude(int row) const {
  if(rowToPersonInclude.count(row) <= 0){
    std::ostringstream os;
    os << "No person from row include: " << row << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
  return rowToPersonInclude[row];
}

int PersonHandler::getRowIncludeFromPerson(Person& person) const {
  std::map<int, Person>::iterator personIterator = rowToPersonInclude.find(person);
  if(personIterator == rowToPersonInclude.end()){
    std::ostringstream os;
    os << "No person from row include: " << row << std::endl;
    const std::string& tmp = os.str();
    throw FileReaderException(tmp.c_str());
  }
  return *personIterator;
}

} /* namespace CuEira */
