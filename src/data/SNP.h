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

  void setCaseAlleleFrequencies(double alleleOneCaseFrequency, double alleleTwoCaseFrequency);
  void setControlAlleleFrequencies(double alleleOneControlFrequency, double alleleTwoControlFrequency);
  void setAllAlleleFrequencies(double alleleOneAllFrequency, double alleleTwoAllFrequency);

  double getAlleleOneCaseFrequency() const;
  double getAlleleTwoCaseFrequency() const;
  double getAlleleOneControlFrequency() const;
  double getAlleleTwoControlFrequency() const;
  double getAlleleOneAllFrequency() const;
  double getAlleleTwoAllFrequency() const;

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

  double alleleOneCaseFrequency;
  double alleleTwoCaseFrequency;
  double alleleOneControlFrequency;
  double alleleTwoControlFrequency;
  double alleleOneAllFrequency;
  double alleleTwoAllFrequency;

  bool caseAlleleHasBeenSet;
  bool controlAlleleHasBeenSet;
  bool allAlleleHasBeenSet;
};

} /* namespace CuEira */

#endif /* SNP_H_ */
