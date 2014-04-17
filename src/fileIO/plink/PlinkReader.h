#ifndef PLINKREADER_H_
#define PLINKREADER_H_

#include <map>
#include <BedReader.h>
#include <BimReader.h>
#include <FamReader.h>
#include <../../container/HostVector.h>
#include <../data/Id.h>
#include <../data/SNP.h>
#include <../data/Person.h>

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PlinkReader {
public:
  PlinkReader(BedReader& bedReader, BimReader& bimReader, FamReader& famReader);
  virtual ~PlinkReader();

  Container::HostVector readSNP(SNP& snp);
  Container::HostVector getOutcomes();
  std::map<Id, Person>& getIdToPersonMap();

  int getNumberOfIndividuals();

private:
  BedReader bedReader;
  BimReader bimReader;
  FamReader famReader;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* PLINKREADER_H_ */
