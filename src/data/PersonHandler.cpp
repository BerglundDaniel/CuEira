#include "PersonHandler.h"

namespace CuEira {

PersonHandler::PersonHandler(std::vector<Person*>* persons) :
    numberOfIndividualsTotal(persons->size()), persons(persons), individualsLocked(false), numberOfIndividualsToInclude(
        0){

  for(auto person : *persons){
    idToPerson.insert(std::pair<Id, Person*>(person->getId(), person));
  }
}

PersonHandler::~PersonHandler(){
  persons->clear();
  delete persons;
}

int PersonHandler::getNumberOfIndividualsTotal() const{
  return numberOfIndividualsTotal;
}

int PersonHandler::getNumberOfIndividualsToInclude() const{
  if(!individualsLocked){
    std::ostringstream os;
    os << "Individuals are not locked so can not access numberOfIndividualsToInclude" << std::endl;
    const std::string& tmp = os.str();
    throw PersonHandlerException(tmp.c_str());
  }

  return numberOfIndividualsToInclude;
}

const std::vector<Person*>& PersonHandler::getPersons() const{
  return *persons;
}

const Person& PersonHandler::getPersonFromId(Id id) const{
  return *idToPerson.at(id);
}

Person& PersonHandler::getPersonFromId(Id id){
  if(individualsLocked){
    std::ostringstream os;
    os << "Individuals are locked in PersonHandler, can't modify them" << std::endl;
    const std::string& tmp = os.str();
    throw PersonHandlerException(tmp.c_str());
  }

  return *idToPerson.at(id);
}

const Person& PersonHandler::getPersonFromRowAll(int rowAll) const{
  if(rowAll > numberOfIndividualsTotal){
    std::ostringstream os;
    os << "Row all larger than number of individuals " << std::endl;
    const std::string& tmp = os.str();
    throw PersonHandlerException(tmp.c_str());
  }
  return *(*persons)[rowAll - 1];
}

int PersonHandler::getRowIncludeFromPerson(const Person& person) const{
  if(!individualsLocked){
    std::ostringstream os;
    os << "Individuals are not locked so can not get row include for person " << person.getId() << std::endl;
    const std::string& tmp = os.str();
    throw PersonHandlerException(tmp.c_str());
  }

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

void PersonHandler::lockIndividuals(){
#ifdef DEBUG
  if(!individualsLocked){
    std::ostringstream os;
    os << "Individuals are already locked." << std::endl;
    const std::string& tmp = os.str();
    throw PersonHandlerException(tmp.c_str());
  }
#endif

  individualsLocked = true;
  int individualNumber = 0;

  for(auto person : *persons){
    if(person->getInclude()){
      individualNumber++;
      personToRowInclude.insert(std::pair<Person*, int>(person, individualNumber));
    }
  }

  numberOfIndividualsToInclude = individualNumber;
}

}
/* namespace CuEira */
