#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <../data/GeneticModel.h>

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
  int getNumberOfStreams();

  /**
   * Get the genetic model
   */
  GeneticModel getGeneticModel();

  /**
   * Get the path to the csv file that contains the environment factors
   */
  std::string getEnvironmentFilePath();

  /**
   * Get the path to the csv file that contains the covariates
   */
  std::string getCovariateFilePath();

  /**
   * Get the name of the column that contains the personds id in the environment file
   */
  std::string getEnvironmentIndividualIdColumnName();

  /**
   * Get the name of the column that contains the personds id in the covariate file
   */
  std::string getCovariateIndividualIdColumnName();

  /**
   * Get the path to the bed file
   */
  std::string getBedFilePath();

  /**
   * Get the path to the fam file
   */
  std::string getFamFilePath();

  /**
   * Get the path to the bim file
   */
  std::string getBimFilePath();

  /**
   * Get the path to the output file
   */
  std::string getOutputFilePath();

  /**
   * Returns true if a covariate file was specified
   */
  bool covariateFileSpecified();

private:
  options::variables_map optionsMap;
  GeneticModel geneticModel;
};

} /* namespace CuEira */

#endif /* CONFIGURATION_H_ */
