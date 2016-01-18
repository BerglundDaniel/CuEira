#include "PersonHandler.h"

namespace CuEira {

PersonHandler::PersonHandler(std::vector<Person*>* persons) :
    numberOfIndividualsTotal(persons->size()), persons(persons), idToPerson(new std::map<Id, Person*>()){

  for(auto person : *persons){
    idToPerson->insert(std::pair<Id, Person*>(person->getId(), person));
  }
}

PersonHandler::~PersonHandler(){
  for(auto person : *persons){
    delete person;
  }
  delete persons;
}

int PersonHandler::getNumberOfIndividualsTotal() const{
  return numberOfIndividualsTotal;
}

PersonHandler::iterator PersonHandler::begin() noexcept{
  return persons->begin();
}

PersonHandler::iterator PersonHandler::end() noexcept{
  return persons->end();
}

const Person& PersonHandler::getPersonFromId(Id id) const{
  return *idToPerson->at(id);
}

Person& PersonHandler::getPersonFromId(Id id){
  return *idToPerson->at(id);
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

Person& PersonHandler::getPersonFromRowAll(int rowAll){
  if(rowAll > numberOfIndividualsTotal){
    std::ostringstream os;
    os << "Row all larger than number of individuals." << std::endl;
    const std::string& tmp = os.str();
    throw PersonHandlerException(tmp.c_str());
  }
  return *(*persons)[rowAll - 1];
}

}
/* namespace CuEira */
