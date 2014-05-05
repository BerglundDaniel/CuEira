#include <ConstructorHelpers.h>

namespace CuEira {
namespace CuEira_Test {

ConstructorHelpers::ConstructorHelpers() {
  srand(time(NULL));
}

ConstructorHelpers::~ConstructorHelpers() {

}

Person ConstructorHelpers::constructPersonInclude(int number) {
  std::ostringstream os;
  os << "Person" << number;
  Id id(os.str());
  Sex sex;
  Phenotype phenotype;

  if(rand() % 2 == 0){
    sex = MALE;
  }else{
    sex = FEMALE;
  }

  if(rand() % 2 == 0){
    phenotype = AFFECTED;
  }else{
    phenotype = UNAFFECTED;
  }

  return Person(id, sex, phenotype);
}

Person ConstructorHelpers::constructPersonNotInclude(int number) {
  std::ostringstream os;
  os << "Person" << number;
  Id id(os.str());
  Sex sex;

  if(rand() % 2 == 0){
    sex = MALE;
  }else{
    sex = FEMALE;
  }

  return Person(id, sex, MISSING);
}

} /* namespace CuEira_Test */
} /* namespace CuEira */
