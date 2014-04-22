#ifndef FAMREADER_H_
#define FAMREADER_H_

#include <map>

#include <HostVector.h>
#include <Id.h>
#include <Person.h>

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class FamReader {
public:
  explicit FamReader(Configuration& configuration);
  virtual ~FamReader();

  Container::HostVector getOutcomes();
  int getNumberOfIndividuals();
  std::map<Id, Person>& getIdToPersonMap();

private:
  Configuration& configuration;
  int numberOfIndividuals;
  map<Id, Person>& idToPersonMap;
  Container::HostVector outcomes;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* FAMREADER_H_ */
