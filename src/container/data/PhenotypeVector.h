#ifndef PHENOTYPEVECTOR_H_
#define PHENOTYPEVECTOR_H_

#include <set>

#include <PhenotypeHandler.h>

namespace CuEira {
namespace Container {

/**
 * This class ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class PhenotypeVector {
public:
  explicit PhenotypeVector(const PhenotypeHandler& phenotypeHandler);
  virtual ~PhenotypeVector();

  virtual void applyMissing(const std::set<int>& personsToSkip);
  virtual void applyNoMissing();

  virtual int getNumberOfIndividualsTotal() const;
  virtual int getNumberOfIndividualsToInclude() const;

protected:
  virtual void copyNonMissingData(const std::set<int>& personsToSkip)=0;

  const int numberOfIndividualsTotal;
  int numberOfIndividualsToInclude;
  bool initialised;
  bool noMissing;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* PHENOTYPEVECTOR_H_ */
