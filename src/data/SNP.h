#ifndef SNP_H_
#define SNP_H_

#include <sstream>
#include <stdexcept>

#include <RiskAllele.h>
#include <Id.h>

namespace CuEira {

/**
 * This class contains information about a column of SNPs, its id and if it should be included in the calculations.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class SNP {
public:
  explicit SNP(Id id, std::string alleleOneName, std::string alleleTwoName, bool include = true);
  virtual ~SNP();

  Id getId();

  bool getInclude();
  void setInclude(bool include);

  std::string getAlleleOneName();
  std::string getAlleleTwoName();

  void setMinorAlleleFrequency(double maf);
  double getMinorAlleleFrequency() const;

  void setRiskAllele(RiskAllele riskAllele);
  RiskAllele getRiskAllele() const;

private:
  Id id;
  bool include;
  std::string alleleOneName;
  std::string alleleTwoName;

  double minorAlleleFrequency;
  bool minorAlleleFrequencyHasBeenSet;

  RiskAllele riskAllele;
  bool riskAlleleHasBeenSet;
};

} /* namespace CuEira */

#endif /* SNP_H_ */
