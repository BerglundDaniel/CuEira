#ifndef CPUMODELHANDLER_H_
#define CPUMODELHANDLER_H_

#include <ModelHandler.h>
#include <DataHandler.h>
#include <InteractionStatistics.h>
#include <Recode.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <SNP.h>
#include <EnvironmentFactor.h>
#include <LogisticRegression.h>
#include <MultipleLogisticRegression.h>

namespace CuEira {
namespace Model {

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuModelHandler: public ModelHandler {
public:
  CpuModelHandler(DataHandler* dataHandler, Container::HostMatrix* covariates);
  virtual ~CpuModelHandler();

  virtual InteractionStatistics* calculateModel();

protected:
  Container::HostMatrix* covariates;

};

} /* namespace Model */
} /* namespace CuEira */

#endif /* CPUMODELHANDLER_H_ */
