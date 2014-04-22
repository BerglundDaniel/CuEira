#ifndef FILEREADEREXCEPTION_H
#define FILEREADEREXCEPTION_H

#include <stdexcept>

/**
 * Exception when there is a problem with reading files
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

class FileReaderException: public std::exception {
public:
  FileReaderException(const char* errMessage) :
      errMessage(errMessage) {

  }

  const char* what() const throw() {
    return errMessage;
  }

private:
  const char* errMessage;
};

#endif // FILEREADEREXCEPTION_H
