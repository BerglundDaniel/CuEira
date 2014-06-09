#ifndef ENVIRONMENTFACTORHANDLEREXCEPTION_H
#define ENVIRONMENTFACTORHANDLEREXCEPTION_H

#include <stdexcept>

/**
 * Exception when there is a problem with EnvironmentFactorHandler
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

class EnvironmentFactorHandlerException: public std::exception {
public:
  EnvironmentFactorHandlerException(const char* errMessage) :
      errMessage(errMessage) {

  }

  const char* what() const throw() {
    return errMessage;
  }

private:
  const char* errMessage;
};

#endif // ENVIRONMENTFACTORHANDLEREXCEPTION_H
