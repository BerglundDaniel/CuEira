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
      "output,o", options::value<std::string>()->required(), "Set output file.")("nstreams,n",
      options::value<int>()->default_value(2), "Set number of streams to use for each GPU. Default 2.")("p",
      options::value<bool>()->zero_tokens(),
      "Use alternative coding for the phenotype, 0 for unaffected and 1 for affected instead of 1 for unaffected and 2 for affected.")(
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
      phenotypeCoding=ZERO_ONE_CODING;
    }
    else{
      phenotypeCoding=ONE_TWO_CODING;
    }
  }

}

Configuration::~Configuration() {

}

int Configuration::getNumberOfStreams() {
  return optionsMap["nstreams"].as<int>();
}

GeneticModel Configuration::getGeneticModel() {
  return geneticModel;
}

std::string Configuration::getBedFilePath() {
  std::ostringstream os;
  os << optionsMap["binary"].as<std::string>() << ".bed";
  return os.str();
}

std::string Configuration::getBimFilePath() {
  std::ostringstream os;
  os << optionsMap["binary"].as<std::string>() << ".bim";
  return os.str();
}

std::string Configuration::getFamFilePath() {
  std::ostringstream os;
  os << optionsMap["binary"].as<std::string>() << ".fam";
  return os.str();
}

std::string Configuration::getEnvironmentFilePath() {
  return optionsMap["enviroment_file"].as<std::string>();
}

std::string Configuration::getCovariateFilePath() {
  return optionsMap["covariate_file"].as<std::string>();
}

std::string Configuration::getEnvironmentIndividualIdColumnName() {
  return optionsMap["environment_id_column"].as<std::string>();
}

std::string Configuration::getCovariateIndividualIdColumnName() {
  return optionsMap["covariate_id_column"].as<std::string>();
}

std::string Configuration::getOutputFilePath() {
  return optionsMap["output"].as<std::string>();
}

bool Configuration::covariateFileSpecified() {
  return optionsMap.count("covariate_file");
}

PhenotypeCoding Configuration::getPhenotypeCoding() {
  return phenotypeCoding;
}

} /* namespace CuEira */
