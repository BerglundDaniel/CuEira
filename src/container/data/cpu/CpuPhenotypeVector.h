#ifndef CPUPHENOTYPEVECTOR_H_
#define CPUPHENOTYPEVECTOR_H_

#include <PhenotypeVector.h>
#include <RegularHostVector.h>
#include <CpuPhenotypeHandler.h>
#include <InvalidState.h>
#include <CpuMissingDataHandler.h>

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
  virtual void applyMissing(const CpuMissingDataHandler& missingDataHandler);

protected:
  const CpuPhenotypeHandler& cpuPhenotypeHandler;
  const RegularHostVector& orgData;
  RegularHostVector* phenotypeExMissing;
};

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CPUPHENOTYPEVECTOR_H_ */
