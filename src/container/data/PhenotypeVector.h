#ifndef PHENOTYPEVECTOR_H_
#define PHENOTYPEVECTOR_H_

#include <PhenotypeHandler.h>
#include <MissingDataHandler.h>

namespace CuEira {
namespace Container {

/**
 * This class ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PhenotypeVector {
public:
  virtual ~PhenotypeVector();

  virtual void applyMissing(const MissingDataHandler& missingDataHandler);
  virtual void applyMissing();

  virtual int getNumberOfIndividualsTotal() const;
  virtual int getNumberOfIndividualsToInclude() const;

protected:
  explicit PhenotypeVector(const PhenotypeHandler& phenotypeHandler);

  const int numberOfIndividualsTotal;
  int numberOfIndividualsToInclude;
  bool initialised;
  bool noMissing;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* PHENOTYPEVECTOR_H_ */
