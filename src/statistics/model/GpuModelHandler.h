#ifndef GPUMODELHANDLER_H_
#define GPUMODELHANDLER_H_

#include <ModelHandler.h>
#include <DataHandler.h>
#include <Statistics.h>
#include <Recode.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <SNP.h>
#include <EnvironmentFactor.h>
#include <LogisticRegressionConfiguration.h>
#include <LogisticRegression.h>
#include <PinnedHostVector.h>
#include <InvalidState.h>
#include <HostToDevice.h>
#include <DeviceToHost.h>

namespace CuEira {
namespace Model {

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class GpuModelHandler: public ModelHandler {
public:
  GpuModelHandler(DataHandler* dataHandler,
      LogisticRegression::LogisticRegressionConfiguration* logisticRegressionConfiguration, const CUDA::HostToDevice& hostToDevice, const CUDA::DeviceToHost& deviceToHost);
  virtual ~GpuModelHandler();

  virtual Statistics* calculateModel();

protected:
  const CUDA::HostToDevice& hostToDevice;
  const CUDA::DeviceToHost& deviceToHost;
  LogisticRegression::LogisticRegressionConfiguration* logisticRegressionConfiguration;
  const int numberOfRows;
  const int numberOfPredictors;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* GPUMODELHANDLER_H_ */
