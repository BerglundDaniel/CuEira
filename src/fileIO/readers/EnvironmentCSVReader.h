#ifndef ENVIRONMENTCSVREADER_H_
#define ENVIRONMENTCSVREADER_H_

#include <string>
#include <utility>
#include <vector>

#include <CSVReader.h>
#include <EnvironmentFactor.h>
#include <FileReaderException.h>
#include <VariableType.h>
#include <Id.h>
#include <PersonHandler.h>
#include <EnvironmentFactorHandler.h>

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class EnvironmentCSVReader: public CSVReader {
public:
  EnvironmentCSVReader(std::string filePath, std::string idColumnName, std::string delim);
  virtual ~EnvironmentCSVReader();

  EnvironmentFactorHandler* readEnvironmentFactorInformation(const PersonHandler& personHandler) const;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* ENVIRONMENTCSVREADER_H_ */
