#ifndef CUDAEXCEPTION_H
#define CUDAEXCEPTION_H

#include <stdexcept>

/**
 * Exception for Cuda errors.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

class CudaException: public std::exception {
public:
  CudaException(const char* errMessage) :
      errMessage(errMessage) {

  }

  const char* what() const throw() {
    return errMessage;
  }

private:
  const char* errMessage;
};

#endif // CUDAEXCEPTION_H
