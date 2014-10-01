#ifndef GPUMODELHANDLER_H_
#define GPUMODELHANDLER_H_

#include <ModelHandler.h>
#include <DataHandler.h>
#include <InteractionStatistics.h>
#include <Recode.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <SNP.h>
#include <EnvironmentFactor.h>
#include <LogisticRegressionConfiguration.h>
#include <LogisticRegression.h>
#include <PinnedHostVector.h>
#include <InvalidState.h>
#include <LogisticRegressionResult.h>
#include <CombinedResultsFactory.h>
#include <CombinedResults.h>
#include <ModelResult.h>
#include <ModelInformation.h>
#include <Model.h>
#include <StatisticModel.h>

namespace CuEira {
namespace Model {

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class GpuModelHandler: public ModelHandler {
public:
  GpuModelHandler(const CombinedResultsFactory& combinedResultsFactory, DataHandler* dataHandler,
      LogisticRegression::LogisticRegressionConfiguration& logisticRegressionConfiguration,
      LogisticRegression::LogisticRegression* logisticRegression);
  virtual ~GpuModelHandler();

  virtual CombinedResults* calculateModel();

protected:
  LogisticRegression::LogisticRegressionConfiguration& logisticRegressionConfiguration;
  LogisticRegression::LogisticRegression* logisticRegression;
  const int numberOfRows;
  const int numberOfPredictors;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* GPUMODELHANDLER_H_ */
