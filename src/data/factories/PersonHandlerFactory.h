#ifndef PERSONHANDLERFACTORY_H_
#define PERSONHANDLERFACTORY_H_

#include <vector>

#include <Person.h>
#include <PersonHandler.h>

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PersonHandlerFactory {
public:
  PersonHandlerFactory();
  virtual ~PersonHandlerFactory();

  virtual PersonHandler* constructPersonHandler(std::vector<Person*>* persons) const;
};

} /* namespace CuEira */

#endif /* PERSONHANDLERFACTORY_H_ */
