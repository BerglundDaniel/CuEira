#ifndef INVALID_STATE_H
#define INVALID_STATE_H

#include <stdexcept>

/**
 * Exception when the state of a class is wrong
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

class InvalidState: public std::exception {
public:
  InvalidState(const char* errMessage) :
      errMessage(errMessage) {

  }

  const char* what() const throw() {
    return errMessage;
  }

private:
  const char* errMessage;
};

#endif // INVALID_STATE_H
