#ifndef MKLEXCEPTION_H
#define MKLEXCEPTION_H

#include <stdexcept>

/**
 * Exception for MKLWrapper
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

class MKLException: public std::exception {
public:
  MKLException(const char* errMessage) :
      errMessage(errMessage) {

  }

  const char* what() const throw() {
    return errMessage;
  }

private:
  const char* errMessage;
};

#endif // MKLEXCEPTION_H
