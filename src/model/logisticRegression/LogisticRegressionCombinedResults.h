#ifndef LOGISTICREGRESSIONCOMBINEDRESULTS_H_
#define LOGISTICREGRESSIONCOMBINEDRESULTS_H_

#include <ostream>

#include <InteractionStatistics.h>
#include <OddsRatioStatistics.h>
#include <Recode.h>
#include <CombinedResults.h>

namespace CuEira {
namespace Model {
namespace LogisticRegression {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LogisticRegressionCombinedResults: public CombinedResults {
  friend std::ostream& operator<<(std::ostream& os,
      const LogisticRegressionCombinedResults& logisticRegressionCombinedResults);
public:
  LogisticRegressionCombinedResults(InteractionStatistics* additiveStatistics,
      OddsRatioStatistics* multiplicativeStatistics, Recode recode);
  virtual ~LogisticRegressionCombinedResults();

  LogisticRegressionCombinedResults(const LogisticRegressionCombinedResults&) = delete;
  LogisticRegressionCombinedResults(LogisticRegressionCombinedResults&&) = delete;
  LogisticRegressionCombinedResults& operator=(const LogisticRegressionCombinedResults&) = delete;
  LogisticRegressionCombinedResults& operator=(LogisticRegressionCombinedResults&&) = delete;

protected:
  LogisticRegressionCombinedResults(); //For the mock
  virtual void toOstream(std::ostream& os) const;

  InteractionStatistics* additiveStatistics;
  OddsRatioStatistics* multiplicativeStatistics;
  Recode recode;
};

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif /* LOGISTICREGRESSIONCOMBINEDRESULTS_H_ */
