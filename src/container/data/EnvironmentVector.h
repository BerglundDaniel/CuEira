#ifndef ENVIRONMENTVECTOR_H_
#define ENVIRONMENTVECTOR_H_

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <HostVector.h>
#include <Recode.h>
#include <StatisticModel.h>
#include <EnvironmentFactor.h>
#include <VariableType.h>
#include <InvalidState.h>
#include <MissingDataHandler.h>

#ifdef CPU
#include <RegularHostVector.h>
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
  FRIEND_TEST(EnvironmentVectorTest, RecodeDifferentOrder);
public:
  EnvironmentVector(const EnvironmentFactor& environmentFactor, const int numberOfIndividualsTotal);
  virtual ~EnvironmentVector();

  virtual const EnvironmentFactor& getEnvironmentFactor() const;
  virtual int getNumberOfIndividualsTotal() const;
  virtual int getNumberOfIndividualsToInclude() const;
  virtual const Container::Vector& getEnvironmentData() const=0;

  virtual void recode(Recode recode, const MissingDataHandler& missingDataHandler)=0;
  virtual void recode(Recode recode)=0;

  EnvironmentVector(const EnvironmentVector&) = delete;
  EnvironmentVector(EnvironmentVector&&) = delete;
  EnvironmentVector& operator=(const EnvironmentVector&) = delete;
  EnvironmentVector& operator=(EnvironmentVector&&) = delete;

private:
  const EnvironmentFactor& environmentFactor;
  const int numberOfIndividualsTotal;
  int numberOfIndividualsToInclude;
  bool initialised;
  bool noMissing;

  Recode currentRecode;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* ENVIRONMENTVECTOR_H_ */
