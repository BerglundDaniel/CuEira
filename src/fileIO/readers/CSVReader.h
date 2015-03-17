#ifndef CSVREADER_H_
#define CSVREADER_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
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
  friend CSVReaderTest;FRIEND_TEST(CSVReaderTest, StoreDataException);
public:
  explicit CSVReader(PersonHandler& personHandler, std::string filePath, std::string idColumnName, std::string delim);
  virtual ~CSVReader();

  virtual int getNumberOfIndividualsTotal() const;
  virtual Container::HostMatrix* readData() const;
  virtual const std::vector<std::string>& getDataColumnNames() const;

protected:
  void storeData(std::vector<std::string> lineSplit, int idColumnNumber, Container::HostMatrix* dataMatrix,
      const int dataRowNumber) const;
  virtual bool rowHasMissingData(const std::vector<std::string>& lineSplit) const;
  bool stringIsEmpty(const std::string& string) const;

  PersonHandler& personHandler;
  const std::string idColumnName;
  int idColumnNumber;
  const std::string delim;
  const std::string filePath;
  int numberOfIndividualsTotal;
  int numberOfDataColumns; //i.e excluding the id column
  std::vector<std::string>* dataColumnNames;

private:
  void readBasicFileInformation();
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* CSVREADER_H_ */
