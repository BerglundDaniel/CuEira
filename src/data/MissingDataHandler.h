#ifndef MISSINGDATAHANDLER_H_
#define MISSINGDATAHANDLER_H_

#include <set>
#include <algorithm>

#include <Vector.h>
#include <HostVector.h>
#include <InvalidState.h>

#ifdef CPU
#include <RegularHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

namespace CuEira {

/*
 * This class ...
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class MissingDataHandler {
public:
  explicit MissingDataHandler(const int numberOfIndividualsTotal);
  virtual ~MissingDataHandler();

  virtual int getNumberOfIndividualsToInclude() const;
  virtual int getNumberOfIndividualsTotal() const;

  virtual void setMissing(const std::set<int>& snpPersonsToSkip, const std::set<int>& envPersonsToSkip);
  virtual Vector* copyNonMissing(const Vector& fromVector) const=0;

private:
  const int numberOfIndividualsTotal;
  int numberOfIndividualsToInclude;
  Container::HostVector* indexesToCopy;
  bool initialised;
};

} /* namespace CuEira */

#endif /* MISSINGDATAHANDLER_H_ */
