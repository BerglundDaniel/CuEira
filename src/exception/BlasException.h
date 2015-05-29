#ifndef BLASEXCEPTION_H
#define BLASEXCEPTION_H

#include <stdexcept>

/**
 * Exception for BlasWrapper
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

class BlasException: public std::exception {
public:
  BlasException(const char* errMessage) :
      errMessage(errMessage){

  }

  const char* what() const throw(){
    return errMessage;
  }

private:
  const char* errMessage;
};

#endif // MKLEXCEPTION_H
