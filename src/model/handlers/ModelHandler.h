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
#include <CombinedResults.h>
#include <CombinedResultsFactory.h>
#include <ModelResult.h>
#include <ModelInformation.h>
#include <Model.h>
#include <DataHandlerState.h>

namespace CuEira {
namespace Model {
class GpuModelHandlerTest;

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Matrix, typename Vector>
class ModelHandler {
  friend GpuModelHandlerTest;
  FRIEND_TEST(GpuModelHandlerTest, Next);
  FRIEND_TEST(GpuModelHandlerTest, NextAndCalculate);
public:
  ModelHandler(const CombinedResultsFactory& combinedResultsFactory, DataHandler<Matrix, Vector>* dataHandler);
  virtual ~ModelHandler();

  virtual DataHandlerState next();
  virtual CombinedResults* calculateModel()=0;

  virtual const ModelInformation& getCurrentModelInformation() const;
  virtual const SNP& getCurrentSNP() const;
  virtual const EnvironmentFactor& getCurrentEnvironmentFactor() const;

  virtual const Container::SNPVector<Vector>& getSNPVector() const;
  virtual const Container::InteractionVector<Vector>& getInteractionVector() const;
  virtual const Container::EnvironmentVector<Vector>& getEnvironmentVector() const;

protected:
  enum State {
    NOT_INITIALISED, INITIALISED_READY, INITIALISED_FULL
  };

  const CombinedResultsFactory& combinedResultsFactory;
  DataHandler<Matrix, Vector>* dataHandler;
  const Vector* environmentData;
  const Vector* snpData;
  const Vector* interactionData;
  const SNP* currentSNP;
  const EnvironmentFactor* currentEnvironmentFactor;
  const SNP* oldSNP;
  const EnvironmentFactor* oldEnvironmentFactor;
  State state;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* MODELHANDLER_H_ */
