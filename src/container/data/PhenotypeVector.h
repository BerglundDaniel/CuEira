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
template<typename Vector>
class PhenotypeVector {
public:
  explicit PhenotypeVector(const PhenotypeHandler<Vector>& phenotypeHandler);
  virtual ~PhenotypeVector();

  virtual int getNumberOfIndividualsTotal() const;
  virtual int getNumberOfIndividualsToInclude() const;
  virtual const Vector& getPhenotypeData() const;

  virtual void applyMissing(const MissingDataHandler<Vector>& missingDataHandler);
  virtual void applyMissing();

protected:
  const PhenotypeHandler<Vector>& phenotypeHandler;
  const int numberOfIndividualsTotal;
  int numberOfIndividualsToInclude;
  bool initialised;
  bool noMissing;

  const Vector& orgData;
  const Vector* phenotypeExMissing;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* PHENOTYPEVECTOR_H_ */
