#ifndef PERSON_H_
#define PERSON_H_

#include <string>
#include <Sex.h>
#include <Id.h>

namespace CuEira {

/**
 * This class contains the information about a person, its id, sex and associated row in the data.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class Person {
public:
  explicit Person(Id id, Sex sex, int rowNumber);
  virtual ~Person();

  Id getId();
  Sex getSex();
  int getRowNumber();

private:
  Id id;
  Sex sex;
  int rowNumber;
};

} /* namespace CuEira */

#endif /* PERSON_H_ */
