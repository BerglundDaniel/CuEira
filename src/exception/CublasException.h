#ifndef CUBLASEXCEPTION_H
#define CUBLASEXCEPTION_H

#include <stdexcept>

/**
 * Exception for Cublas errors.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

class CublasException: public std::exception {
public:
  CublasException(const char* errMessage) :
      errMessage(errMessage) {

  }

  const char* what() const throw() {
    return errMessage;
  }

private:
  const char* errMessage;
};

#endif // CUBLASEXCEPTION_H
