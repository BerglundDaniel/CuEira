#ifndef DATAFILESREADER_H_
#define DATAFILESREADER_H_

#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <set>
#include <algorithm>
#include <iterator>

#include <CSVReader.h>
#include <BedReader.h>
#include <BimReader.h>
#include <Configuration.h>
#include <HostVector.h>
#include <SNPVector.h>
#include <HostMatrix.h>
#include <SNP.h>
#include <EnvironmentFactor.h>
#include <PersonHandler.h>
#include <FileReaderException.h>

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataFilesReader {
public:
  explicit DataFilesReader(PersonHandler* personHandler, BedReader* bedReader, BimReader* bimReader,
      CSVReader* csvReader);
  virtual ~DataFilesReader();

  virtual Container::HostMatrix* readCSV() const;
  virtual const std::vector<std::string>& getCSVDataColumnNames() const;
  virtual std::vector<SNP*>* readSNPInformation() const;
  virtual Container::SNPVector* readSNP(SNP& snp);

  virtual const PersonHandler& getPersonHandler() const;

private:
  PersonHandler *personHandler;
  BedReader* bedReader;
  BimReader* bimReader;
  CSVReader* csvReader;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* DATAFILESREADER_H_ */
