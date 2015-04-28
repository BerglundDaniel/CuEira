#ifndef INVALID_ARGUMENT_H
#define INVALID_ARGUMENT_H

#include <stdexcept>

/**
 * Exception when an argument is wrong
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

class InvalidArgument: public std::exception {
public:
  InvalidArgument(const char* errMessage) :
      errMessage(errMessage) {

  }

  const char* what() const throw() {
    return errMessage;
  }

private:
  const char* errMessage;
};

#endif // INVALID_ARGUMENT_H
