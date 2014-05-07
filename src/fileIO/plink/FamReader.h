#ifndef FAMREADER_H_
#define FAMREADER_H_

#include <fstream>
#include <sstream>
#include <string>

#include <Id.h>
#include <Sex.h>
#include <Person.h>
#include <PersonHandler.h>
#include <Phenotype.h>
#include <PhenotypeCoding.h>
#include <Configuration.h>
#include <FileReaderException.h>

namespace CuEira {

namespace CuEira_Test {
class FamReaderTest;
}

namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class FamReader {
  friend CuEira_Test::FamReaderTest;
public:
  explicit FamReader(const Configuration& configuration, PersonHandler& personHandler);
  virtual ~FamReader();

  const PersonHandler& getPersonHandler() const;

private:
  Phenotype stringToPhenotype(std::string phenotypeString) const;
  Sex stringToSex(std::string sexString) const;

  const Configuration& configuration;
  PersonHandler& personHandler;
  const std::string famFileStr;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* FAMREADER_H_ */
