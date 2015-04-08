#ifndef ENVIRONMENTVECTOR_H_
#define ENVIRONMENTVECTOR_H_

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <HostVector.h>
#include <Recode.h>
#include <StatisticModel.h>
#include <EnvironmentFactorHandler.h>
#include <EnvironmentFactor.h>
#include <VariableType.h>
#include <InvalidState.h>
#include <MissingDataHandler.h>

namespace CuEira {
namespace Container {
class EnvironmentVectorTest;

/**
 * This class ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Vector>
class EnvironmentVector {
  friend EnvironmentVectorTest;
  FRIEND_TEST(EnvironmentVectorTest, ConstructAndGet);
  FRIEND_TEST(EnvironmentVectorTest, RecodeNonBinary);
  FRIEND_TEST(EnvironmentVectorTest, RecodeBinary);
  FRIEND_TEST(EnvironmentVectorTest, RecodeDifferentOrder);
public:
  EnvironmentVector(const EnvironmentFactorHandler<Vector>& environmentFactorHandler);
  virtual ~EnvironmentVector();

  virtual const EnvironmentFactor& getEnvironmentFactor() const;
  virtual int getNumberOfIndividualsTotal() const;
  virtual int getNumberOfIndividualsToInclude() const;
  virtual const Vector& getEnvironmentData() const;
  virtual Vector& getEnvironmentData();

  virtual void recode(Recode recode);
  virtual void recode(Recode recode, const MissingDataHandler<Vector>& missingDataHandler);

  EnvironmentVector(const EnvironmentVector&) = delete;
  EnvironmentVector(EnvironmentVector&&) = delete;
  EnvironmentVector& operator=(const EnvironmentVector&) = delete;
  EnvironmentVector& operator=(EnvironmentVector&&) = delete;

protected:
  virtual void recodeProtective()=0;
  virtual void recodeAllRisk()=0;

  const EnvironmentFactor& environmentFactor;
  const int numberOfIndividualsTotal;
  int numberOfIndividualsToInclude;
  bool initialised;
  bool noMissing;

  const Vector& originalData;
  Vector* envExMissing;

  Recode currentRecode;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* ENVIRONMENTVECTOR_H_ */
