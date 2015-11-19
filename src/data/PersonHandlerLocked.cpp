#include "PersonHandlerLocked.h"

namespace CuEira {

PersonHandlerLocked::PersonHandlerLocked(PersonHandler& personHandler) :
    numberOfIndividualsTotal(personHandler.numberOfIndividualsTotal), idToPerson(personHandler.idToPerson), persons(
        new std::vector<const Person*>(numberOfIndividualsTotal)){
  personHandler.idToPerson = nullptr;
  int individualNumber = 0;

  auto personsIter = persons->begin();
  for(auto person : *(personHandler.persons)){
    *personsIter = person;
    ++personsIter;

    if(person->getInclude()){
      individualNumber++;
      personToRowInclude.insert(std::pair<Person*, int>(person, individualNumber));
    }
  }

  delete personHandler.persons;
  personHandler.persons = nullptr;
  numberOfIndividualsToInclude = individualNumber;
}

PersonHandlerLocked::~PersonHandlerLocked(){

}

int PersonHandlerLocked::getNumberOfIndividualsTotal() const{
  return numberOfIndividualsTotal;
}

int PersonHandlerLocked::getNumberOfIndividualsToInclude() const{
  return numberOfIndividualsToInclude;
}

PersonHandlerLocked::iterator PersonHandlerLocked::begin() const noexcept{
  return persons->begin();
}

PersonHandlerLocked::iterator PersonHandlerLocked::end() const noexcept{
  return persons->end();
}

const Person& PersonHandlerLocked::getPersonFromId(Id id) const{
  return *idToPerson.at(id);
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
