#ifndef CSVREADER_H_
#define CSVREADER_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>

#include <FileReaderException.h>
#include <Person.h>
#include <PersonHandler.h>
#include <Id.h>
#include <HostVector.h>
#include <HostMatrix.h>

#ifdef CPU
#include <lapackpp/gmd.h>
#include <lapackpp/lavd.h>
#include <LapackppHostMatrix.h>
#else
#include <PinnedHostMatrix.h>
#endif

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CSVReader {
public:
  explicit CSVReader(std::string filePath, std::string idColumnName, std::string delim,
      const PersonHandler& personHandler);
  virtual ~CSVReader();

  int getNumberOfColumns(); //Not including id
  int getNumberOfRows(); //Not including header
  std::vector<std::string> getDataColumnHeaders();
  Container::HostMatrix& getData();
  Container::HostVector& getData(std::string column);

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
