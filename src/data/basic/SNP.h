#ifndef SNP_H_
#define SNP_H_

#include <vector>
#include <ostream>

#include <RiskAllele.h>
#include <Id.h>
#include <InvalidState.h>
#include <SNPIncludeExclude.h>

namespace CuEira {

/**
 * This class contains information about a column of SNPs, its id and if it should be included in the calculations.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class SNP {
  friend std::ostream& operator<<(std::ostream& os, const SNP& snp);
public:
  explicit SNP(Id id, std::string alleleOneName, std::string alleleTwoName, unsigned int position,
      SNPIncludeExclude includeExclude = INCLUDE);
  virtual ~SNP();

  Id getId() const;

  bool shouldInclude() const;
  const std::vector<SNPIncludeExclude>& getInclude() const;
  void setInclude(SNPIncludeExclude includeExclude);

  unsigned int getPosition() const;
  std::string getAlleleOneName() const;
  std::string getAlleleTwoName() const;

  void setRiskAllele(RiskAllele riskAllele);
  RiskAllele getRiskAllele() const;
  RiskAllele getProtectiveAllele() const;

  bool operator<(const SNP& otherSNP) const;
  bool operator==(const SNP& otherSNP) const;

  SNP(const SNP&) = delete;
  SNP(SNP&&) = delete;
  SNP& operator=(const SNP&) = delete;
  SNP& operator=(SNP&&) = delete;

private:
  Id id;
  std::vector<SNPIncludeExclude>* includeExcludeVector;
  const std::string alleleOneName;
  const std::string alleleTwoName;
  const unsigned int position;

  RiskAllele riskAllele;
  bool riskAlleleHasBeenSet;
};

} /* namespace CuEira */

#endif /* SNP_H_ */
