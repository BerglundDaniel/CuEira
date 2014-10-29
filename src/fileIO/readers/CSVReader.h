#ifndef CSVREADER_H_
#define CSVREADER_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <boost/algorithm/string.hpp>
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <FileReaderException.h>
#include <Person.h>
#include <PersonHandler.h>
#include <Id.h>
#include <HostVector.h>
#include <HostMatrix.h>

#ifdef CPU
#include <RegularHostMatrix.h>
#else
#include <PinnedHostMatrix.h>
#endif

namespace CuEira {
namespace FileIO {
class CSVReaderTest;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CSVReader {
  friend CSVReaderTest;
  FRIEND_TEST(CSVReaderTest, StoreDataException);
public:
  explicit CSVReader(std::string filePath, std::string idColumnName, std::string delim);
  virtual ~CSVReader();

  std::pair<Container::HostMatrix*, std::vector<std::string>*>* readData(const PersonHandler& personHandler) const;

protected:
  void storeData(std::vector<std::string> lineSplit, int idColumnNumber, Container::HostMatrix* dataMatrix, unsigned int dataRowNumber) const;

  const std::string idColumnName;
  const std::string delim;
  const std::string filePath;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* CSVREADER_H_ */
