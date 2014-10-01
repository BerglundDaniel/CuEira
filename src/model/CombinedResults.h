#ifndef COMBINEDRESULTS_H_
#define COMBINEDRESULTS_H_

#include <ostream>

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
  CombinedResults();
  virtual ~CombinedResults();

  CombinedResults(const CombinedResults&) = delete;
  CombinedResults(CombinedResults&&) = delete;
  CombinedResults& operator=(const CombinedResults&) = delete;
  CombinedResults& operator=(CombinedResults&&) = delete;

protected:
  virtual void toOstream(std::ostream& os) const=0;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* COMBINEDRESULTS_H_ */
