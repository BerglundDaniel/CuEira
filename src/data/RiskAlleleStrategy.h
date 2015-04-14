#ifndef RISKALLELESTRATEGY_H_
#define RISKALLELESTRATEGY_H_

#include <vector>

#include <RiskAllele.h>
#include <AlleleStatistics.h>

namespace CuEira {

/*
 * This class ...
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class RiskAlleleStrategy {
public:
  explicit RiskAlleleStrategy();
  virtual ~RiskAlleleStrategy();

  virtual RiskAllele calculateRiskAllele(const AlleleStatistics& alleleStatistics) const;

  RiskAlleleStrategy(const RiskAlleleStrategy&) = delete;
  RiskAlleleStrategy(RiskAlleleStrategy&&) = delete;
  RiskAlleleStrategy& operator=(const RiskAlleleStrategy&) = delete;
  RiskAlleleStrategy& operator=(RiskAlleleStrategy&&) = delete;

private:

};

} /* namespace CuEira */

#endif /* RISKALLELESTRATEGY_H_ */
