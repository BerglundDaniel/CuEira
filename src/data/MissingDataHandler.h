#ifndef MISSINGDATAHANDLER_H_
#define MISSINGDATAHANDLER_H_

#include <set>
#include <algorithm>

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
template<typename Vector>
class MissingDataHandler {
public:
  explicit MissingDataHandler(const int numberOfIndividualsTotal);
  virtual ~MissingDataHandler();

  virtual int getNumberOfIndividualsToInclude() const;
  virtual int getNumberOfIndividualsTotal() const;

  virtual void setMissing(const std::set<int>& snpPersonsToSkip);
  virtual void copyNonMissing(const Vector& fromVector, Vector& toVector) const=0;

protected:
  const int numberOfIndividualsTotal;
  int numberOfIndividualsToInclude;
  bool initialised;

#ifdef CPU //FIXME template this?
  Container::RegularHostVector* indexesToCopy;
#else
  Container::PinnedHostVector* indexesToCopy;
#endif
};

} /* namespace CuEira */

#endif /* MISSINGDATAHANDLER_H_ */
