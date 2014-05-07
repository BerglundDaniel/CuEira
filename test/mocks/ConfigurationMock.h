#ifndef CONFIGURATIONMOCK_H_
#define CONFIGURATIONMOCK_H_

#include <gmock/gmock.h>
#include <string>
#include <iostream>

#include <Configuration.h>
#include <GeneticModel.h>
#include <PhenotypeCoding.h>

namespace CuEira {

class ConfigurationMock : public Configuration{
public:
  ConfigurationMock():Configuration(){

  }

  virtual ~ConfigurationMock(){

  }

  MOCK_CONST_METHOD0(getNumberOfStreams, int());
  MOCK_CONST_METHOD0(getGeneticModel, GeneticModel());
  MOCK_CONST_METHOD0(getEnvironmentFilePath, std::string());
  MOCK_CONST_METHOD0(getCovariateFilePath, std::string());
  MOCK_CONST_METHOD0(getEnvironmentIndividualIdColumnName, std::string());
  MOCK_CONST_METHOD0(getCovariateIndividualIdColumnName, std::string());
  MOCK_CONST_METHOD0(getBedFilePath, std::string());
  MOCK_CONST_METHOD0(getFamFilePath, std::string());
  MOCK_CONST_METHOD0(getBimFilePath, std::string());
  MOCK_CONST_METHOD0(getOutputFilePath, std::string());
  MOCK_CONST_METHOD0(covariateFileSpecified, bool());
  MOCK_CONST_METHOD0(getPhenotypeCoding, PhenotypeCoding());
  MOCK_CONST_METHOD0(excludeSNPsWithNegativePosition, bool());
  MOCK_CONST_METHOD0(getMinorAlleleFrequencyThreshold, double());
  MOCK_CONST_METHOD0(getEnvironmentDelimiter, std::string());
  MOCK_CONST_METHOD0(getCovariateDelimiter, std::string());
};

} /* namespace CuEira */

#endif /* CONFIGURATIONMOCK_H_ */
