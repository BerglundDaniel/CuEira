#ifndef SNPVECTOR_H_
#define SNPVECTOR_H_

#include <vector>
#include <ostream>
#include <set>

#include <SNP.h>
#include <Recode.h>
#include <HostVector.h>
#include <GeneticModel.h>
#include <RiskAllele.h>
#include <StatisticModel.h>
#include <InvalidState.h>

namespace CuEira {
namespace Container {

/**
 * This class ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Vector>
class SNPVector {
public:
  /**
   * Construct a SNPVector
   */
  explicit SNPVector(SNP& snp, GeneticModel geneticModel, const Vector* snpOrgExMissing,
      const std::set<int>* snpMissingData);

  virtual ~SNPVector();

  virtual int getNumberOfIndividualsToInclude() const;
  virtual const SNP& getAssociatedSNP() const;
  virtual const Vector& getSNPData() const;
  virtual Vector& getSNPData();
  virtual bool hasMissing() const;
  virtual const std::set<int>& getMissing() const;

  virtual void recode(Recode recode);

  SNPVector(const SNPVector&) = delete;
  SNPVector(SNPVector&&) = delete;
  SNPVector& operator=(const SNPVector&) = delete;
  SNPVector& operator=(SNPVector&&) = delete;

protected:
  virtual void doRecode(int snpToRisk[3])=0;

  SNP& snp;
  const Vector* snpOrgExMissing;
  const std::set<int>* snpMissingData;
  const int numberOfIndividualsToInclude;
  Vector* snpRecodedExMissing;
  bool initialised;
  bool noMissing;

  const GeneticModel originalGeneticModel;
  GeneticModel currentGeneticModel;
  const RiskAllele originalRiskAllele;
  Recode currentRecode;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* SNPVECTOR_H_ */
