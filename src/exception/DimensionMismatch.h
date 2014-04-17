#ifndef DIMENSIONMISMATCH_H
#define DIMENSIONMISMATCH_H

#include <stdexcept>

/**
 * Exception when provided dimensions doesn't match the requirements.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

class DimensionMismatch: public std::exception {
public:
  DimensionMismatch(const char* errMessage) :
      errMessage(errMessage) {

  }

  const char* what() const throw() {
    return errMessage;
  }

private:
  const char* errMessage;
};

#endif // DIMENSIONMISMATCH_H
