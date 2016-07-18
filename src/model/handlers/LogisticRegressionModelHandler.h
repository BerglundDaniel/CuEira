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
#include <AdditiveInteractionModel.h>
#include <MultiplicativeInteractionModel.h>

namespace CuEira {
namespace Model {
namespace LogisticRegression {

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Matrix, typename Vector>
class LogisticRegressionModelHandler: public ModelHandler<Matrix, Vector> {
public:
  LogisticRegressionModelHandler(const CombinedResultsFactory& combinedResultsFactory,
      DataHandler<Matrix, Vector>* dataHandler,
      CuEira::Model::LogisticRegression::LogisticRegressionConfiguration& logisticRegressionConfiguration,
      LogisticRegression* logisticRegression, AdditiveInteractionModel<Vector>* additiveInteractionModel,
      MultiplicativeInteractionModel<Vector>* multiplicativeInteractionModel);
  virtual ~LogisticRegressionModelHandler();

  virtual CombinedResults* calculateModel();

protected:
  CuEira::Model::LogisticRegression::LogisticRegressionConfiguration& logisticRegressionConfiguration;
  LogisticRegression* logisticRegression;
  AdditiveInteractionModel<Vector>* additiveInteractionModel;
  MultiplicativeInteractionModel<Vector>* multiplicativeInteractionModel;
  const int numberOfRows;
  const int numberOfPredictors;
};

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif /* LOGISTICREGRESSIONMODELHANDLER_H_ */
