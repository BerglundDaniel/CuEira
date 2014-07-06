#ifndef SNPVECTOR_H_
#define SNPVECTOR_H_

#define ALLELE_ONE_CASE_POSITION 0
#define ALLELE_TWO_CASE_POSITION 1
#define ALLELE_ONE_CONTROL_POSITION 2
#define ALLELE_TWO_CONTROL_POSITION 3
#define ALLELE_ONE_ALL_POSITION 4
#define ALLELE_TWO_ALL_POSITION 5
#define ABSOLUTE_FREQUENCY_THRESHOLD 5

#include <vector>
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
  /**
   * Constructs a normal SNPVector
   */
  SNPVector(SNP& snp, GeneticModel geneticModel, const std::vector<int>* originalSNPData, const std::vector<int>* numberOfAlleles,
      const std::vector<double>* alleleFrequencies);

  /**
   * Constructs a SNPVector that only holds the allele frequencies. Used when the SNP shouldn't be included.
   */
  SNPVector(SNP& snp, const std::vector<int>* numberOfAlleles, const std::vector<double>* alleleFrequencies,
      int numberOfIndividualsToInclude);

  virtual ~SNPVector();

  virtual int getNumberOfIndividualsToInclude() const;
  virtual const std::vector<int>& getOrginalData() const;
  virtual const Container::HostVector& getRecodedData() const;
  virtual const SNP& getAssociatedSNP() const;
  virtual void recode(Recode recode);
  virtual void applyStatisticModel(StatisticModel statisticModel, const HostVector& interactionVector);
  virtual const std::vector<int>& getAlleleNumbers() const;
  virtual const std::vector<double>& getAlleleFrequencies() const;

private:
  void recodeAllRisk();
  void recodeSNPProtective();
  void recodeInteractionProtective();
  void doRecode();
  RiskAllele invertRiskAllele(RiskAllele riskAllele);

  const int numberOfIndividualsToInclude;
  SNP& snp;
  const std::vector<int>* numberOfAlleles;
  const std::vector<double>* alleleFrequencies;
  RiskAllele currentRiskAllele;
  GeneticModel currentGeneticModel;
  const RiskAllele originalRiskAllele;
  const GeneticModel originalGeneticModel;
  const std::vector<int>* originalSNPData;
  Container::HostVector* modifiedSNPData;
  Recode currentRecode;
  bool onlyFrequencies;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* SNPVECTOR_H_ */
