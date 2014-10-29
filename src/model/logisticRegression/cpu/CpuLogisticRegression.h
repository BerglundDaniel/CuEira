#ifndef CPULOGISTICREGRESSION_H_
#define CPULOGISTICREGRESSION_H_

#include <math.h>
#include <algorithm>
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <LogisticRegression.h>
#include <LogisticRegressionResult.h>
#include <CpuLogisticRegressionConfiguration.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <RegularHostMatrix.h>
#include <RegularHostVector.h>

namespace CuEira {
namespace Model {
namespace LogisticRegression {
namespace CPU {
class CpuLogisticRegressionTest;

using namespace CuEira::Container;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuLogisticRegression: public LogisticRegression {
  friend CpuLogisticRegressionTest;
  FRIEND_TEST(CpuLogisticRegressionTest, calcuateProbabilites);
  FRIEND_TEST(CpuLogisticRegressionTest, calculateScores);
  FRIEND_TEST(CpuLogisticRegressionTest, calculateInformationMatrix);
  FRIEND_TEST(CpuLogisticRegressionTest, calculateLogLikelihood);
public:
  CpuLogisticRegression(CpuLogisticRegressionConfiguration* cpuLogisticRegressionConfiguration);
  virtual ~CpuLogisticRegression();

  /**
   * Calculate the model
   */
  virtual LogisticRegressionResult* calculate();

  CpuLogisticRegression(const CpuLogisticRegression&) = delete;
  CpuLogisticRegression(CpuLogisticRegression&&) = delete;
  CpuLogisticRegression& operator=(const CpuLogisticRegression&) = delete;
  CpuLogisticRegression& operator=(CpuLogisticRegression&&) = delete;

protected:
  CpuLogisticRegression(); //For the mock

private:
  void calcuateProbabilites(const HostMatrix& predictors, const HostVector& betaCoefficents, HostVector& probabilites,
      HostVector& workVectorNx1);
  void calculateScores(const HostMatrix& predictors, const HostVector& outcomes, const HostVector& probabilites,
      HostVector& scores, HostVector& workVectorNx1);
  void calculateInformationMatrix(const HostMatrix& predictors, const HostVector& probabilites,
      HostVector& workVectorNx1, HostMatrix& informationMatrix, HostMatrix& workMatrixNxM);
  void calculateLogLikelihood(const HostVector& outcomes, const HostVector& probabilites, PRECISION& logLikelihood);

  LogisticRegressionConfiguration* cpuLogisticRegressionConfiguration;
  const HostVector* outcomes;

  //NOTE The config class owns this memory
  const HostMatrix* predictors;
  Container::HostVector* probabilites;
  Container::HostMatrix* workMatrixNxM;
  Container::HostVector* workVectorNx1;
  const Container::HostVector* defaultBetaCoefficents;
};

} /* namespace CPU */
} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif /* CPULOGISTICREGRESSION_H_ */
