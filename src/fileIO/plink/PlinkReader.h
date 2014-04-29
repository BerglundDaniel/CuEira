#ifndef PLINKREADER_H_
#define PLINKREADER_H_

#include <map>
#include <vector>

#include <BedReader.h>
#include <BimReader.h>
#include <FamReader.h>
#include <HostVector.h>
#include <Id.h>
#include <SNP.h>
#include <Person.h>
#include <PersonHandler.h>

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

  Container::HostVector& readSNP(SNP& snp);
  const PersonHandler& getPersonHandler() const;

private:
  BedReader bedReader;
  BimReader bimReader;
  FamReader famReader;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* PLINKREADER_H_ */
