#ifndef LOGISTICREGRESSIONMODELHANDLER_H_
#define LOGISTICREGRESSIONMODELHANDLER_H_

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
#include <InvalidState.h>
#include <LogisticRegressionResult.h>
#include <CombinedResultsFactory.h>
#include <CombinedResults.h>
#include <ModelResult.h>
#include <ModelInformation.h>
#include <Model.h>
#include <StatisticModel.h>

#include <PinnedHostVector.h> //TMP thing TODO based on problems with set function

namespace CuEira {
namespace Model {
namespace LogisticRegression {

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LogisticRegressionModelHandler: public ModelHandler {
public:
  LogisticRegressionModelHandler(const CombinedResultsFactory& combinedResultsFactory, DataHandler* dataHandler,
      CuEira::Model::LogisticRegression::LogisticRegressionConfiguration& logisticRegressionConfiguration,
      LogisticRegression* logisticRegression);
  virtual ~LogisticRegressionModelHandler();

  virtual CombinedResults* calculateModel();

protected:
  CuEira::Model::LogisticRegression::LogisticRegressionConfiguration& logisticRegressionConfiguration;
  LogisticRegression* logisticRegression;
  const int numberOfRows;
  const int numberOfPredictors;
};

} /* namespace CudaLogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif /* LOGISTICREGRESSIONMODELHANDLER_H_ */
