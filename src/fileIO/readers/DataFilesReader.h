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
#include <CovariatesHandler.h>
#include <CovariatesHandlerFactory.h>
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
  explicit DataFilesReader(CovariatesHandlerFactory* covariatesHandlerFactory, PersonHandler* personHandler,
      BedReader* bedReader, BimReader* bimReader, EnvironmentCSVReader* environmentCSVReader,
      CSVReader* covariateCSVReader);
  explicit DataFilesReader(PersonHandler* personHandler, BedReader* bedReader, BimReader* bimReader,
      EnvironmentCSVReader* environmentCSVReader);
  virtual ~DataFilesReader();

  virtual CovariatesHandler* readCovariates() const;
  virtual std::vector<SNP*>* readSNPInformation() const;
  virtual EnvironmentFactorHandler* readEnvironmentFactorInformation() const;
  virtual Container::SNPVector* readSNP(SNP& snp);

  virtual const PersonHandler& getPersonHandler() const;

private:
  bool useCovariates;
  CovariatesHandlerFactory* covariatesHandlerFactory;
  PersonHandler *personHandler;
  BedReader* bedReader;
  BimReader* bimReader;
  EnvironmentCSVReader* environmentCSVReader;
  CSVReader* covariateCSVReader;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* DATAFILESREADER_H_ */
