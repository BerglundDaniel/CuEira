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
#include <lapackpp/gmd.h>
#include <LapackppHostMatrix.h>
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
  FRIEND_TEST(CSVReaderTest, GetDataException);
public:
  explicit CSVReader(std::string filePath, std::string idColumnName, std::string delim,
      const PersonHandler& personHandler);
  virtual ~CSVReader();

  int getNumberOfColumns() const; //Not including id
  int getNumberOfRows() const; //Not including header
  const std::vector<std::string>& getDataColumnHeaders() const; //Not including id
  const Container::HostMatrix& getData() const;
  const Container::HostVector& getData(std::string column) const;

private:
  void storeData(std::vector<std::string> lineSplit);

  const PersonHandler& personHandler;
  const std::string delim;
  const std::string filePath;
  const std::string idColumnName;
  const int numberOfIndividualsToInclude;
  const int numberOfIndividualsTotal;
  int numberOfColumns; //Not including id
  int numberOfRows; //Not including header
  int idColumnNumber;
  std::vector<std::string> dataColumnNames;

#ifdef CPU
  Container::LapackppHostMatrix* dataMatrix;
#else
  Container::PinnedHostMatrix dataMatrix;
#endif
};

#endif /* CSVREADER_H_ */

} /* namespace FileIO */
} /* namespace CuEira */
