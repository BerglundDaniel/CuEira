#ifndef MODELHANDLER_H_
#define MODELHANDLER_H_

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <DataHandler.h>
#include <Recode.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <SNP.h>
#include <EnvironmentFactor.h>
#include <ModelState.h>
#include <CombinedResults.h>
#include <CombinedResultsFactory.h>
#include <ModelResult.h>
#include <ModelInformation.h>
#include <Model.h>

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
  ModelHandler(const CombinedResultsFactory& combinedResultsFactory, DataHandler* dataHandler);
  virtual ~ModelHandler();

  virtual ModelInformation* next();
  virtual CombinedResults* calculateModel()=0;

  virtual const SNP& getCurrentSNP() const;
  virtual const EnvironmentFactor& getCurrentEnvironmentFactor() const;

  virtual const Container::SNPVector& getSNPVector() const;
  virtual const Container::InteractionVector& getInteractionVector() const;
  virtual const Container::EnvironmentVector& getEnvironmentVector() const;

protected:
  enum State{
    NOT_INITIALISED, INITIALISED_READY, INITIALISED_FULL
  };

  const CombinedResultsFactory& combinedResultsFactory;
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