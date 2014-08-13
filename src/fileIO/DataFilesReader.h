#ifndef DATAFILESREADER_H_
#define DATAFILESREADER_H_

#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <CSVReader.h>
#include <EnvironmentCSVReader.h>
#include <BimReader.h>
#include <FamReader.h>
#include <Configuration.h>
#include <HostVector.h>
#include <SNPVector.h>
#include <HostMatrix.h>
#include <SNP.h>
#include <EnvironmentFactor.h>
#include <PersonHandler.h>
#include <EnvironmentFactorHandler.h>
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
  explicit DataFilesReader(BimReader* bimReader, FamReader* famReader, EnvironmentCSVReader* environmentCSVReader,
      CSVReader* covariateCSVReader);
  explicit DataFilesReader(BimReader* bimReader, FamReader* famReader, EnvironmentCSVReader* environmentCSVReader);
  virtual ~DataFilesReader();

  std::pair<Container::HostMatrix*, std::vector<std::string>* >* readCovariates(const PersonHandler& personHandler) const;
  PersonHandler* readPersonInformation() const;
  std::vector<SNP*>* readSNPInformation() const;
  EnvironmentFactorHandler* readEnvironmentFactorInformation(const PersonHandler& personHandler) const;

private:
  bool useCovariates;
  BimReader* bimReader;
  FamReader* famReader;
  EnvironmentCSVReader* environmentCSVReader;
  CSVReader* covariateCSVReader;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* DATAFILESREADER_H_ */
