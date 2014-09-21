#ifndef COMBINEDRESULTS_H_
#define COMBINEDRESULTS_H_

#include <ostream>

#include <InteractionStatistics.h>
#include <Recode.h>

namespace CuEira {
namespace Model {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CombinedResults {
  friend std::ostream& operator<<(std::ostream& os, const CombinedResults& combinedResults);
public:
  CombinedResults(InteractionStatistics* interactionStatistics, Recode recode);
  virtual ~CombinedResults();

  CombinedResults(const CombinedResults&) = delete;
  CombinedResults(CombinedResults&&) = delete;
  CombinedResults& operator=(const CombinedResults&) = delete;
  CombinedResults& operator=(CombinedResults&&) = delete;

protected:
  virtual void toOstream(std::ostream& os) const;

private:
  InteractionStatistics* interactionStatistics;
  Recode recode;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* COMBINEDRESULTS_H_ */
