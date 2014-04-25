#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include <GeneticModel.h>
#include <PhenotypeCoding.h>

namespace CuEira {

namespace options = boost::program_options;

/**
 * This class handles the command line arguments and stores the options for CuEira.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class Configuration {
public:
  Configuration(int argc, char* argv[]);
  virtual ~Configuration();

  /**
   * Get the number of streams to use for each GPU
   */
  int getNumberOfStreams() const;

  /**
   * Get the genetic model
   */
  GeneticModel getGeneticModel() const;

  /**
   * Get the path to the csv file that contains the environment factors
   */
  std::string getEnvironmentFilePath() const;

  /**
   * Get the path to the csv file that contains the covariates
   */
  std::string getCovariateFilePath() const;

  /**
   * Get the name of the column that contains the personds id in the environment file
   */
  std::string getEnvironmentIndividualIdColumnName() const;

  /**
   * Get the name of the column that contains the personds id in the covariate file
   */
  std::string getCovariateIndividualIdColumnName() const;

  /**
   * Get the path to the bed file
   */
  std::string getBedFilePath() const;

  /**
   * Get the path to the fam file
   */
  std::string getFamFilePath() const;

  /**
   * Get the path to the bim file
   */
  std::string getBimFilePath() const;

  /**
   * Get the path to the output file
   */
  std::string getOutputFilePath() const;

  /**
   * Returns true if a covariate file was specified
   */
  bool covariateFileSpecified() const;

  /**
   * Returns the specified coding for the phenotypes in the fam file
   */
  PhenotypeCoding getPhenotypeCoding() const;

  /**
   * Returns true if SNPs with negative position should be excluded.
   */
  bool excludeSNPsWithNegativePosition() const;

  /**
   * Returns the threshold for minor allele frequency.
   */
  double getMinorAlleleFrequencyThreshold() const;

private:
  options::variables_map optionsMap;
  GeneticModel geneticModel;
  PhenotypeCoding phenotypeCoding;
};

} /* namespace CuEira */

#endif /* CONFIGURATION_H_ */
