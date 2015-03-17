#ifndef CPUPHENOTYPEVECTOR_H_
#define CPUPHENOTYPEVECTOR_H_

#include <set>

#include <PhenotypeVector.h>
#include <RegularHostVector.h>
#include <CpuPhenotypeHandler.h>
#include <InvalidState.h>

namespace CuEira {
namespace Container {
namespace CPU {

using namespace CuEira::CPU;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuPhenotypeVector: public PhenotypeVector {
public:
  CpuPhenotypeVector(const CpuPhenotypeHandler& cpuPhenotypeHandler);
  virtual ~CpuPhenotypeVector();

  virtual const RegularHostVector& getPhenotypeData() const;

protected:
  virtual void copyNonMissingData(const std::set<int>& personsToSkip);

  const CpuPhenotypeHandler& cpuPhenotypeHandler;
  const RegularHostVector& orgData;
  RegularHostVector* phenotypeExMissing;
};

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CPUPHENOTYPEVECTOR_H_ */
