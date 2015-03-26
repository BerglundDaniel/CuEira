#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <GeneticModel.h>
#include <PhenotypeCoding.h>
#include <HostVector.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

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
   * Get the number of GPUs to use
   */
  virtual int getNumberOfGPUs() const;

  /**
   * Was the number of GPUs option used
   */
  virtual bool isNumberOfGPUsSet() const;

  /**
   * Get the genetic model
   */
  virtual GeneticModel getGeneticModel() const;

  /**
   * Get the path to the csv file that contains the covariates and environment factors
   */
  virtual std::string getCSVFilePath() const;

  /**
   * Get the name of the column that contains the personds id in the environment file
   */
  virtual std::string getCSVIdColumnName() const;

  /**
   * Get the name of the column that contains the personds id in the environment file
   */
  virtual std::string getEnvironmentColumnName() const;

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
   * Get the specified coding for the phenotypes in the fam file
   */
  virtual PhenotypeCoding getPhenotypeCoding() const;

  /**
   * Returns true if SNPs with negative position should be excluded.
   */
  virtual bool excludeSNPsWithNegativePosition() const;

  /**
   * Get the threshold for minor allele frequency.
   */
  virtual double getMinorAlleleFrequencyThreshold() const;

  /**
   * Get the delimiter for the csv file
   */
  virtual std::string getCSVDelimiter() const;

  /**
   * Get the max number of iterations for the logistic regression
   */
  virtual int getNumberOfMaxLRIterations() const;

  /**
   * Get the convergence threshold for the logistic regression
   */
  virtual double getLRConvergenceThreshold() const;

  /**
   * Get the threshold for the cell cell counts of the contingency table
   */
  virtual int getCellCountThreshold() const;

  Configuration(const Configuration&) = delete;
  Configuration(Configuration&&) = delete;
  Configuration& operator=(const Configuration&) = delete;
  Configuration& operator=(Configuration&&) = delete;

protected:
  Configuration(); //Used for the mock of this class

private:
  options::variables_map optionsMap;
  GeneticModel geneticModel;
  PhenotypeCoding phenotypeCoding;
};

} /* namespace CuEira */

#endif /* CONFIGURATION_H_ */
