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
#include <EnvironmentCSVReader.h>
#include <BedReader.h>
#include <BimReader.h>
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
  explicit DataFilesReader(PersonHandler* personHandler, BedReader* bedReader, BimReader* bimReader,
      EnvironmentCSVReader* environmentCSVReader, CSVReader* covariateCSVReader);
  explicit DataFilesReader(PersonHandler* personHandler, BedReader* bedReader, BimReader* bimReader,
      EnvironmentCSVReader* environmentCSVReader);
  virtual ~DataFilesReader();

  virtual Container::HostMatrix* readCovariates() const;
  virtual std::vector<SNP*>* readSNPInformation() const;
  virtual EnvironmentFactorHandler* readEnvironmentFactorInformation() const;
  virtual std::pair<const AlleleStatistics*, Container::SNPVector*>* readSNP(SNP& snp);

  virtual const PersonHandler& getPersonHandler() const;

private:
  bool useCovariates;
  BedReader* bedReader;
  BimReader* bimReader;
  EnvironmentCSVReader* environmentCSVReader;
  CSVReader* covariateCSVReader;
  PersonHandler *personHandler;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* DATAFILESREADER_H_ */
