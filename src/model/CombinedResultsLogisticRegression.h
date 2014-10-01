#ifndef COMBINEDRESULTSLOGISTICREGRESSION_H_
#define COMBINEDRESULTSLOGISTICREGRESSION_H_

#include <ostream>

#include <InteractionStatistics.h>
#include <OddsRatioStatistics.h>
#include <Recode.h>
#include <CombinedResults.h>

namespace CuEira {
namespace Model {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CombinedResultsLogisticRegression: public CombinedResults {
  friend std::ostream& operator<<(std::ostream& os,
      const CombinedResultsLogisticRegression& combinedResultsLogisticRegression);
public:
  CombinedResultsLogisticRegression(InteractionStatistics* additiveStatistics,
      OddsRatioStatistics* multiplicativeStatistics, Recode recode);
  virtual ~CombinedResultsLogisticRegression();

  CombinedResultsLogisticRegression(const CombinedResultsLogisticRegression&) = delete;
  CombinedResultsLogisticRegression(CombinedResultsLogisticRegression&&) = delete;
  CombinedResultsLogisticRegression& operator=(const CombinedResultsLogisticRegression&) = delete;
  CombinedResultsLogisticRegression& operator=(CombinedResultsLogisticRegression&&) = delete;

protected:
  CombinedResultsLogisticRegression(); //For the mock
  virtual void toOstream(std::ostream& os) const;

  InteractionStatistics* additiveStatistics;
  OddsRatioStatistics* multiplicativeStatistics;
  Recode recode;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* COMBINEDRESULTSLOGISTICREGRESSION_H_ */
