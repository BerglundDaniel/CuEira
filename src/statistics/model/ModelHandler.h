#ifndef MODELHANDLER_H_
#define MODELHANDLER_H_

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <DataHandler.h>
#include <Statistics.h>
#include <Recode.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <SNP.h>
#include <EnvironmentFactor.h>
#include <StatisticsFactory.h>

namespace CuEira {
namespace Model {
class GpuModelHandlerTest;

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ModelHandler {
  friend GpuModelHandlerTest;
  FRIEND_TEST(GpuModelHandlerTest, Next);
  FRIEND_TEST(GpuModelHandlerTest, NextAndCalculate);
public:
  ModelHandler(const StatisticsFactory& statisticsFactory, DataHandler* dataHandler);
  virtual ~ModelHandler();

  bool next();
  virtual Statistics* calculateModel()=0;

protected:
  enum State{
    NOT_INITIALISED, INITIALISED_READY, INITIALISED_FULL
  };

  const StatisticsFactory& statisticsFactory;
  DataHandler* dataHandler;
  const Container::HostVector * environmentData;
  const Container::HostVector * snpData;
  const Container::HostVector * interactionData;
  const SNP* currentSNP;
  const EnvironmentFactor* currentEnvironmentFactor;
  const SNP* oldSNP;
  const EnvironmentFactor* oldEnvironmentFactor;
  State state;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* MODELHANDLER_H_ */
