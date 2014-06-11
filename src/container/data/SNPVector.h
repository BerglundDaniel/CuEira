#ifndef SNPVECTOR_H_
#define SNPVECTOR_H_

#include <vector>
#include <stdexcept>
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <SNP.h>
#include <Recode.h>
#include <HostVector.h>
#include <GeneticModel.h>
#include <RiskAllele.h>
#include <StatisticModel.h>

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
  FRIEND_TEST(SNPVectorTest, DoRecodeDominantAlleleOne);
  FRIEND_TEST(SNPVectorTest, DoRecodeDominantAlleleTwo);
  FRIEND_TEST(SNPVectorTest, DoRecodeRecessiveAlleleOne);
  FRIEND_TEST(SNPVectorTest, DoRecodeRecessiveAlleleTwo);
  FRIEND_TEST(SNPVectorTest, InvertRiskAllele);
public:
  SNPVector(std::vector<int>* originalSNPData, SNP& snp, GeneticModel geneticModel);
  virtual ~SNPVector();

  virtual int getNumberOfIndividualsToInclude() const;
  virtual const std::vector<int>& getOrginalData() const;
  virtual const Container::HostVector& getRecodedData() const;
  virtual const SNP& getAssociatedSNP() const;
  virtual void recode(Recode recode);
  virtual void applyStatisticModel(StatisticModel statisticModel, const HostVector& interactionVector);

private:
  void recodeAllRisk();
  void recodeSNPProtective();
  void recodeInteractionProtective();
  void doRecode();
  RiskAllele invertRiskAllele(RiskAllele riskAllele);

  const int numberOfIndividualsToInclude;
  SNP& snp;
  RiskAllele currentRiskAllele;
  GeneticModel currentGeneticModel;
  const RiskAllele originalRiskAllele;
  const GeneticModel originalGeneticModel;
  std::vector<int>* originalSNPData;
  Container::HostVector* modifiedSNPData;
  Recode currentRecode;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* SNPVECTOR_H_ */
