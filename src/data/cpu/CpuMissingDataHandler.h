#ifndef CPUMISSINGDATAHANDLER_H_
#define CPUMISSINGDATAHANDLER_H_

#include <MissingDataHandler.h>
#include <HostVector.h>
#include <RegularHostVector.h>

namespace CuEira {
namespace CPU {

/*
 * This class ...
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuMissingDataHandler: public MissingDataHandler {
public:
  explicit CpuMissingDataHandler(const int numberOfIndividualsTotal);
  virtual ~CpuMissingDataHandler();

  virtual Container::HostVector* copyNonMissing(const Container::HostVector& fromVector) const;

protected:

};

} /* namespace CPU */
} /* namespace CuEira */

#endif /* CPUMISSINGDATAHANDLER_H_ */
