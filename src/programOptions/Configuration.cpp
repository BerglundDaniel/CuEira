#include "Configuration.h"

namespace CuEira {

namespace options = boost::program_options;

Configuration::Configuration(int argc, char* argv[]) {
  // Declare the supported options
  options::options_description description("Program usage:");
  description.add_options()("help,h", "Produce help message.")
  //("seed", options::value<int>()->default_value(1), "Set the seed. Default 1")
  ("model,m", options::value<std::string>()->default_value("dominant"),
      "The genetic model type to use(ie dominant or recessive). Default: dominant.")("binary,b",
      options::value<std::string>()->required(), "Name of file in plink binary format")("environment_file,e",
      options::value<std::string>()->required(), "Set the csv file with the environmental variables.")(
      "environment_id_column,x", options::value<std::string>()->required(),
      "Set the name of the column in the enviromental file that holds the person ids.")("covariate_file,c",
      options::value<std::string>(), "Set the csv file with covariates.")("covariate_id_column,z",
      options::value<std::string>(), "Set the name of the column in the covariates file that holds the person ids.")(
      "output,o", options::value<std::string>()->required(), "Set output file.")
  //("nstreams,n",options::value<int>()->default_value(2), "Set number of streams to use for each GPU. Default 2.")
  ("maf,m", options::value<double>()->default_value(0.05),
      "Set the threshold for minor allele frequency(MAF) in range 0 to 1. Any SNPs with MAF below the threshold will be excluded from the analysis. Default 0.05.")(
      "p", options::value<bool>()->zero_tokens(),
      "Use alternative coding for the phenotype, 0 for unaffected and 1 for affected instead of 1 for unaffected and 2 for affected.")(
      "exclude", options::value<bool>()->zero_tokens(), "Exclude SNPs with negative position from the analysis.")(
      "version,v", "Print the version number.");

  options::store(options::parse_command_line(argc, argv, description), optionsMap);

  if(optionsMap.count("help")){
    std::cerr << description << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  options::notify(optionsMap);

  if(optionsMap.count("version")){
    std::cerr << "Version " << CuEira_VERSION_MAJOR << "." << CuEira_VERSION_MINOR << std::endl;
    std::exit(EXIT_SUCCESS);
  }

  if(optionsMap.count("covariate_file")){
    if(!optionsMap.count("covariate_file")){
      throw std::invalid_argument(
          "If a covariate file is specified you must also provided the name of the column with the person ids.");
    }
  }

  if(optionsMap.count("model")){
    std::string geneticModelStr = optionsMap["model"].as<std::string>();
    boost::to_lower(geneticModelStr);
    if(geneticModelStr == "dominant"){
      geneticModel = DOMINANT;
    }else if(geneticModelStr == "recessive"){
      geneticModel = RECESSIVE;
    }else{
      throw std::invalid_argument("Invalid genetic model argument");
    }
  }

  if(optionsMap.count("p")){
    if(optionsMap["p"].as<bool>()){
      phenotypeCoding = ZERO_ONE_CODING;
    }else{
      phenotypeCoding = ONE_TWO_CODING;
    }
  }else{
    phenotypeCoding = ONE_TWO_CODING;
  }

  if(optionsMap["maf"].as<double>() < 0 || optionsMap["maf"].as<double>() > 1){
    throw std::invalid_argument("Minor allele frequency has to be between 0 and 1.");
  }

}

Configuration::~Configuration() {

}

Configuration::Configuration() {

}

int Configuration::getNumberOfStreams() const {
  //return optionsMap["nstreams"].as<int>();
  return 3;
}

GeneticModel Configuration::getGeneticModel() const {
  return geneticModel;
}

std::string Configuration::getBedFilePath() const {
  std::ostringstream os;
  os << optionsMap["binary"].as<std::string>() << ".bed";
  return os.str();
}

std::string Configuration::getBimFilePath() const {
  std::ostringstream os;
  os << optionsMap["binary"].as<std::string>() << ".bim";
  return os.str();
}

std::string Configuration::getFamFilePath() const {
  std::ostringstream os;
  os << optionsMap["binary"].as<std::string>() << ".fam";
  return os.str();
}

std::string Configuration::getEnvironmentFilePath() const {
  return optionsMap["environment_file"].as<std::string>();
}

std::string Configuration::getCovariateFilePath() const {
  return optionsMap["covariate_file"].as<std::string>();
}

std::string Configuration::getEnvironmentIndividualIdColumnName() const {
  return optionsMap["environment_id_column"].as<std::string>();
}

std::string Configuration::getCovariateIndividualIdColumnName() const {
  return optionsMap["covariate_id_column"].as<std::string>();
}

std::string Configuration::getOutputFilePath() const {
  return optionsMap["output"].as<std::string>();
}

bool Configuration::covariateFileSpecified() const {
  return optionsMap.count("covariate_file");
}

PhenotypeCoding Configuration::getPhenotypeCoding() const {
  return phenotypeCoding;
}

bool Configuration::excludeSNPsWithNegativePosition() const {
  if(optionsMap.count("exclude")){
    if(optionsMap["exclude"].as<bool>()){
      return true;
    }else{
      return false;
    }
  }else{
    return false;
  }
}

double Configuration::getMinorAlleleFrequencyThreshold() const {
  return optionsMap["maf"].as<double>();
}

std::string Configuration::getEnvironmentDelimiter() const {
  return "\t "; //FIXME set as option
}

std::string Configuration::getCovariateDelimiter() const {
  return "\t "; //FIXME set as option
}

int Configuration::getNumberOfMaxLRIterations() const {
  return 500; //FIXME set as option
}

double Configuration::getLRConvergenceThreshold() const {
  return 1e-5; //FIXME set as option
}

StatisticModel Configuration::getStatisticModel() const {
  return ADDITIVE; //FIXME set as option
}

} /* namespace CuEira */
