#ifndef ENVIRONMENTCSVREADER_H_
#define ENVIRONMENTCSVREADER_H_

#include <string>

#include <CSVReader.h>
#include <EnvironmentFactor.h>
#include <FileReaderException.h>
#include <VariableType.h>
#include <Id.h>
#include <PersonHandler.h>

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class EnvironmentCSVReader: public CSVReader {
public:
  EnvironmentCSVReader(std::string filePath, std::string idColumnName, std::string delim,
      const PersonHandler& personHandler);
  virtual ~EnvironmentCSVReader();

  const Container::HostVector& getData(EnvironmentFactor& environmentFactor) const;
  const std::vector<EnvironmentFactor*>& getEnvironmentFactorInformation() const;

private:
  std::vector<EnvironmentFactor*>* environmentFactors;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* ENVIRONMENTCSVREADER_H_ */
