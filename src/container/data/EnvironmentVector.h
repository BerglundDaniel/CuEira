#ifndef ENVIRONMENTVECTOR_H_
#define ENVIRONMENTVECTOR_H_

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <HostVector.h>
#include <Recode.h>
#include <StatisticModel.h>
#include <EnvironmentFactor.h>
#include <EnvironmentFactorHandler.h>
#include <VariableType.h>

#ifdef CPU
#include <lapackpp/lavd.h>
#include <LapackppHostVector.h>
#else
#include <PinnedHostVector.h>
#endif

namespace CuEira {
namespace Container {
class EnvironmentVectorTest;

/**
 * This class ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class EnvironmentVector {
  friend EnvironmentVectorTest;
  FRIEND_TEST(EnvironmentVectorTest, ConstructAndGet);
  FRIEND_TEST(EnvironmentVectorTest, RecodeNonBinary);
  FRIEND_TEST(EnvironmentVectorTest, RecodeBinary);
public:
  EnvironmentVector(const EnvironmentFactorHandler& environmentHandler, EnvironmentFactor& environmentFactor);
  virtual ~EnvironmentVector();

  virtual void switchEnvironmentFactor(EnvironmentFactor& environmentFactor);
  virtual int getNumberOfIndividualsToInclude() const;
  virtual const Container::HostVector& getRecodedData() const;
  virtual void recode(Recode recode);
  virtual void applyStatisticModel(StatisticModel statisticModel, const HostVector& interactionVector);
  virtual const EnvironmentFactor& getCurrentEnvironmentFactor() const;

private:
  void recodeAllRisk();
  void recodeEnvironmentProtective();
  void recodeInteractionProtective();
  void doRecode();

  const EnvironmentFactorHandler& environmentHandler;
  const HostVector * originalData;
  int numberOfIndividualsToInclude;
  HostVector* recodedData;
  Recode currentRecode;
  EnvironmentFactor& environmentFactor;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* ENVIRONMENTVECTOR_H_ */
