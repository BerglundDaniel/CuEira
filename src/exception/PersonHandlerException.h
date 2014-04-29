#ifndef PERSONHANDLEREXCEPTION_H
#define PERSONHANDLEREXCEPTION_H

#include <stdexcept>

/**
 * Exception when there is a problem with PersonHandler
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

class PersonHandlerException: public std::exception {
public:
  PersonHandlerException(const char* errMessage) :
      errMessage(errMessage) {

  }

  const char* what() const throw() {
    return errMessage;
  }

private:
  const char* errMessage;
};

#endif // PERSONHANDLEREXCEPTION_H
