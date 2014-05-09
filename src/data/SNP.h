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
  explicit SNP(Id id, std::string alleleOneName, std::string alleleTwoName, unsigned int position, bool include = true);
  virtual ~SNP();

  Id getId() const;

  bool getInclude() const;
  void setInclude(bool include);

  unsigned int getPosition() const;
  std::string getAlleleOneName() const;
  std::string getAlleleTwoName() const;

  void setMinorAlleleFrequency(double maf);
  double getMinorAlleleFrequency() const;

  void setRiskAllele(RiskAllele riskAllele);
  RiskAllele getRiskAllele() const;

private:
  Id id;
  bool include;
  const std::string alleleOneName;
  const std::string alleleTwoName;
  const unsigned int position;

  double minorAlleleFrequency;
  bool minorAlleleFrequencyHasBeenSet;

  RiskAllele riskAllele;
  bool riskAlleleHasBeenSet;
};

} /* namespace CuEira */

#endif /* SNP_H_ */
