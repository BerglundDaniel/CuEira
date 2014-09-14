#ifndef SNPVECTOR_H_
#define SNPVECTOR_H_

#include <vector>
#include <ostream>
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
class SNPVectorFactoryTest;

/**
 * This class ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class SNPVector {
  friend std::ostream& operator<<(std::ostream& os, const Container::SNPVector& snpVector);
  friend SNPVectorTest;
  friend SNPVectorFactoryTest;
  FRIEND_TEST(SNPVectorTest, DoRecodeDominantAlleleOne);
  FRIEND_TEST(SNPVectorTest, DoRecodeDominantAlleleTwo);
  FRIEND_TEST(SNPVectorTest, DoRecodeRecessiveAlleleOne);
  FRIEND_TEST(SNPVectorTest, DoRecodeRecessiveAlleleTwo);
  FRIEND_TEST(SNPVectorTest, InvertRiskAllele);
  FRIEND_TEST(SNPVectorFactoryTest, ConstructSNPVector);
public:
  /**
   * Construct a SNPVector
   */
  SNPVector(SNP& snp, GeneticModel geneticModel, const std::vector<int>* originalSNPData);

  virtual ~SNPVector();

  virtual int getNumberOfIndividualsToInclude() const;
  virtual const std::vector<int>& getOrginalData() const;
  virtual const Container::HostVector& getRecodedData() const;
  virtual const SNP& getAssociatedSNP() const;
  virtual void recode(Recode recode);
  virtual void applyStatisticModel(StatisticModel statisticModel, const HostVector& interactionVector);

  SNPVector(const SNPVector&) = delete;
  SNPVector(SNPVector&&) = delete;
  SNPVector& operator=(const SNPVector&) = delete;
  SNPVector& operator=(SNPVector&&) = delete;

protected:
  SNPVector(SNP& snp); //For the mock

private:
  void recodeAllRisk();
  void recodeSNPProtective();
  void recodeInteractionProtective();
  void doRecode();
  RiskAllele invertRiskAllele(RiskAllele riskAllele);

  const int numberOfIndividualsToInclude;
  SNP& snp;
  GeneticModel currentGeneticModel;
  const RiskAllele originalRiskAllele;
  const GeneticModel originalGeneticModel;
  const std::vector<int>* originalSNPData;
  Container::HostVector* modifiedSNPData;
  Recode currentRecode;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* SNPVECTOR_H_ */
