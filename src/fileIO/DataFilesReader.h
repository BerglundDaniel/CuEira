#ifndef DATAFILESREADER_H_
#define DATAFILESREADER_H_

#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <CSVReader.h>
#include <../programOptions/Configuration.h>
#include <../container/HostVector.h>
#include <../container/HostMatrix.h>
#include <../data/SNP.h>
#include <../data/EnvironmentFactor.h>
#include <plink/PlinkReader.h>

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class DataFilesReader {
public:
  explicit DataFilesReader(PlinkReader& plinkReader, CSVReader& environmentCSVReader,
      CSVReader& covariateCSVReader);
  virtual ~DataFilesReader();

  Container::HostVector readSNP(SNP& snp);
  Container::HostVector getEnvironmentFactor(EnvironmentFactor& environmentFactor);
  Container::HostMatrix getCovariates();
  Container::HostVector getOutcomes();

  int getNumberOfIndividuals();
  int getNumberOfCovariates();
  int getNumberOfEnvironmentFactors();

private:
  PlinkReader& plinkReader;
  CSVReader& environmentCSVReader;
  CSVReader& covariateCSVReader;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* DATAFILESREADER_H_ */
