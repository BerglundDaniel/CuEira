#ifndef SNPVECTOR_H_
#define SNPVECTOR_H_

#include <vector>
#include <stdexcept>

#include <SNP.h>
#include <Recode.h>
#include <HostVector.h>
#include <InvalidState.h>
#include <GeneticModel.h>
#include <RiskAllele.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

namespace CuEira {
namespace Container {
class SNPVectorTest;

/**
 * This class ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class SNPVector {
  friend SNPVectorTest;
  //FRIEND_TEST(SNPVectorTest, ConstructorCheckMode);
public:
  SNPVector(const std::vector<int>* originalSNPData, SNP& snp, GeneticModel geneticModel);
  virtual ~SNPVector();

  const std::vector<int>* getOrginalData() const;
  const Container::HostVector* getRecodedData() const;
  Recode getRecode() const;
  void recode(Recode recode);

private:
  void recodeAllRisk();
  void recodeSNPProtective();
  void recodeInteractionProtective();
  void doRecode();
  RiskAllele invertRiskAllele(RiskAllele riskAllele);

  const int numberOfIndividualsToInclude;
  SNP& snp;
  Recode currentRecode;
  RiskAllele currentRiskAllele;
  GeneticModel currentGeneticModel;
  const RiskAllele originalRiskAllele;
  const GeneticModel originalGeneticModel;
  const std::vector<int>* originalSNPData;
  Container::HostVector* modifiedSNPData;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* SNPVECTOR_H_ */
