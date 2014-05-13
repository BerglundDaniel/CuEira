#ifndef DATAFILESREADER_H_
#define DATAFILESREADER_H_

#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <CSVReader.h>
#include <EnvironmentCSVReader.h>
#include <Configuration.h>
#include <HostVector.h>
#include <SNPVector.h>
#include <HostMatrix.h>
#include <SNP.h>
#include <EnvironmentFactor.h>
#include <PlinkReader.h>
#include <PersonHandler.h>

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataFilesReader {
public:
  explicit DataFilesReader(PlinkReader* plinkReader, EnvironmentCSVReader* environmentCSVReader, CSVReader* covariateCSVReader);
  virtual ~DataFilesReader();

  Container::SNPVector* readSNP(SNP& snp) const;
  const Container::HostVector& getEnvironmentFactor(EnvironmentFactor& environmentFactor) const;
  const Container::HostMatrix& getCovariates() const;
  const PersonHandler& getPersonHandler() const;
  std::vector<SNP*> getSNPInformation();

  int getNumberOfCovariates() const;
  int getNumberOfEnvironmentFactors() const;

private:
  PlinkReader* plinkReader;
  EnvironmentCSVReader* environmentCSVReader;
  CSVReader* covariateCSVReader;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* DATAFILESREADER_H_ */
