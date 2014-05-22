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
  virtual int getNumberOfStreams() const;

  /**
   * Get the genetic model
   */
  virtual GeneticModel getGeneticModel() const;

  /**
   * Get the path to the csv file that contains the environment factors
   */
  virtual std::string getEnvironmentFilePath() const;

  /**
   * Get the path to the csv file that contains the covariates
   */
  virtual std::string getCovariateFilePath() const;

  /**
   * Get the name of the column that contains the personds id in the environment file
   */
  virtual std::string getEnvironmentIndividualIdColumnName() const;

  /**
   * Get the name of the column that contains the personds id in the covariate file
   */
  virtual std::string getCovariateIndividualIdColumnName() const;

  /**
   * Get the path to the bed file
   */
  virtual std::string getBedFilePath() const;

  /**
   * Get the path to the fam file
   */
  virtual std::string getFamFilePath() const;

  /**
   * Get the path to the bim file
   */
  virtual std::string getBimFilePath() const;

  /**
   * Get the path to the output file
   */
  virtual std::string getOutputFilePath() const;

  /**
   * Returns true if a covariate file was specified
   */
  virtual bool covariateFileSpecified() const;

  /**
   * Returns the specified coding for the phenotypes in the fam file
   */
  virtual PhenotypeCoding getPhenotypeCoding() const;

  /**
   * Returns true if SNPs with negative position should be excluded.
   */
  virtual bool excludeSNPsWithNegativePosition() const;

  /**
   * Returns the threshold for minor allele frequency.
   */
  virtual double getMinorAlleleFrequencyThreshold() const;

  /**
   * Returns the delimiter for the environment csv file
   */
  virtual std::string getEnvironmentDelimiter() const;

  /**
   * Returns the delimiter for the covariate csv file
   */
  virtual std::string getCovariateDelimiter() const;

  /**
   * Returns the max number of iterations for the logistic regression
   */
  virtual int getNumberOfMaxLRIterations() const;

  /**
   * Returns the convergence threshold for the logistic regression
   */
  virtual double getLRConvergenceThreshold() const;

protected:
  Configuration(); //Used for the mock of this class

private:
  options::variables_map optionsMap;
  GeneticModel geneticModel;
  PhenotypeCoding phenotypeCoding;
};

} /* namespace CuEira */

#endif /* CONFIGURATION_H_ */
