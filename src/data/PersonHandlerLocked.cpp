#include "PersonHandlerLocked.h"

namespace CuEira {

PersonHandlerLocked::PersonHandlerLocked(PersonHandler& personHandler) :
    numberOfIndividualsTotal(personHandler.numberOfIndividualsTotal), idToPerson(personHandler.idToPerson), persons(
        personHandler.persons){
  personHandler.idToPerson = nullptr;
  personHandler.persons = nullptr;
  int individualNumber = 0;

  for(auto person : *persons){
    if(person->getInclude()){
      individualNumber++;
      personToRowInclude.insert(std::pair<Person*, int>(person, individualNumber));
    }
  }

  numberOfIndividualsToInclude = individualNumber;
}

PersonHandlerLocked::~PersonHandlerLocked(){
  for(auto person : *persons){
    delete person;
  }
  delete persons;
  delete idToPerson;
}

int PersonHandlerLocked::getNumberOfIndividualsTotal() const{
  return numberOfIndividualsTotal;
}

int PersonHandlerLocked::getNumberOfIndividualsToInclude() const{
  return numberOfIndividualsToInclude;
}

PersonHandlerLocked::const_iterator PersonHandlerLocked::begin() const noexcept{
  return persons->cbegin();
}

PersonHandlerLocked::const_iterator PersonHandlerLocked::end() const noexcept{
  return persons->cend();
}

const Person& PersonHandlerLocked::getPersonFromId(Id id) const{
  return *(idToPerson->at(id));
}

const Person& PersonHandler::getPersonFromRowAll(int rowAll) const{
  if(rowAll > numberOfIndividualsTotal){
    std::ostringstream os;
    os << "Row all larger than number of individuals." << std::endl;
    const std::string& tmp = os.str();
    throw PersonHandlerException(tmp.c_str());
  }
  return *(*persons)[rowAll - 1];
}

int PersonHandlerLocked::getRowIncludeFromPerson(const Person& person) const{
  std::map<const Person*, int, pointerLess<Person> >::const_iterator iterator = personToRowInclude.find(&person);

  if(iterator == personToRowInclude.end()){
    std::ostringstream os;
    os << "Cannot find person in map." << std::endl;
    const std::string& tmp = os.str();
    throw PersonHandlerException(tmp.c_str());
  }else{
    return iterator->second;
  }
}

} /* namespace CuEira */
