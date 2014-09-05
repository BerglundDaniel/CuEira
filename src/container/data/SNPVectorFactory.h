#ifndef SNPVECTORFACTORY_H_
#define SNPVECTORFACTORY_H_

#include <vector>
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <SNPVector.h>
#include <SNP.h>
#include <Configuration.h>
#include <GeneticModel.h>
#include <RiskAllele.h>

namespace CuEira {
namespace Container {
class SNPVectorFactoryTest;

/**
 * This class constructs SNPVectors and makes checks on the data to see if the SNP should be included or not.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class SNPVectorFactory {
  friend SNPVectorFactoryTest;
  ///FIXME
  FRIEND_TEST(SNPVectorFactoryTest, AlleleFrequencies);
  FRIEND_TEST(SNPVectorFactoryTest, RiskAllele_Equal_Case_Control);
  FRIEND_TEST(SNPVectorFactoryTest, RiskAllele_Equal_Case_Control1_larger_Control2);
  FRIEND_TEST(SNPVectorFactoryTest, RiskAllele_Equal_Case_Control2_larger_Control1);
  FRIEND_TEST(SNPVectorFactoryTest, RiskAllele_Case1_Larger_Case2_Case1_Larger_Control1);
  FRIEND_TEST(SNPVectorFactoryTest, RiskAllele_Case1_Larger_Case2_Case1_Smaller_Control1);
  FRIEND_TEST(SNPVectorFactoryTest, RiskAllele_Case2_Larger_Case1_Case2_Larger_Control2);
  FRIEND_TEST(SNPVectorFactoryTest, RiskAllele_Case2_Larger_Case1_Case2_Smaller_Control2);
  FRIEND_TEST(SNPVectorFactoryTest, SNPInclude_MissingData);
  FRIEND_TEST(SNPVectorFactoryTest, SNPInclude_True);
  FRIEND_TEST(SNPVectorFactoryTest, SNPInclude_ToLowAbsFreq_1);
  FRIEND_TEST(SNPVectorFactoryTest, SNPInclude_ToLowAbsFreq_2);
  FRIEND_TEST(SNPVectorFactoryTest, SNPInclude_ToLowAbsFreq_3);
  FRIEND_TEST(SNPVectorFactoryTest, SNPInclude_ToLowAbsFreq_4);
  FRIEND_TEST(SNPVectorFactoryTest, SNPInclude_ToLowMAF_Equal);
  FRIEND_TEST(SNPVectorFactoryTest, SNPInclude_ToLowMAF_1Larger2);
  FRIEND_TEST(SNPVectorFactoryTest, SNPInclude_ToLowMAF_2Larger1);
public:
  SNPVectorFactory(const Configuration& configuration);
  virtual ~SNPVectorFactory();

  virtual SNPVector* constructSNPVector(SNP& snp, const std::vector<int>* originalSNPData) const;

private:
  const Configuration& configuration;
  const GeneticModel geneticModel;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* SNPVECTORFACTORY_H_ */
