#ifndef CSVREADER_H_
#define CSVREADER_H_

#include <map>
#include <string>
#include <stdexcept>
#include <Person.h>
#include <Id.h>

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CSVReader {
public:
  explicit CSVReader(std::string filePath, std::string idColumnName, std::map<Id, Person>& idToPersonMap);
  virtual ~CSVReader();

  int getNumberOfColumns(); //Not inclding id
  int getNumberOfRows(); //Not including header
  HostMatrix getData();

private:
  std::map<Id, Person>& idToPersonMap;
};

#endif /* CSVREADER_H_ */

} /* namespace FileIO */
} /* namespace CuEira */
