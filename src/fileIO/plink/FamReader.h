#ifndef FAMREADER_H_
#define FAMREADER_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <HostVector.h>
#include <Id.h>
#include <Person.h>
#include <Phenotype.h>
#include <PhenotypeCoding.h>
#include <Configuration.h>
#include <FileReaderException.h>

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class FamReader {
public:
  explicit FamReader(const Configuration& configuration);
  virtual ~FamReader();

  const Container::HostVector& getOutcomes() const;
  int getNumberOfIndividuals() const;
  const std::vector<Person*>& getPersons() const;

private:
  const Configuration& configuration;
  int numberOfIndividuals;
  std::vector<Person*> persons;
  Container::HostVector outcomes;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* FAMREADER_H_ */
