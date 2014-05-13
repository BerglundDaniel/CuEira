#ifndef CONSTRUCTORHELPERS_H_
#define CONSTRUCTORHELPERS_H_

#include <sstream>
#include <string>
#include <stdlib.h>
#include <time.h>

#include <Id.h>
#include <Sex.h>
#include <Person.h>
#include <Phenotype.h>

namespace CuEira {
namespace CuEira_Test {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ConstructorHelpers {
public:
  ConstructorHelpers();
  virtual ~ConstructorHelpers();

  Person* constructPersonInclude(int number);
  Person* constructPersonNotInclude(int number);

  Person* constructPersonInclude(int number, Phenotype phenotype);

private:

};

} /* namespace CuEira_Test */
} /* namespace CuEira */

#endif /* CONSTRUCTORHELPERS_H_ */
