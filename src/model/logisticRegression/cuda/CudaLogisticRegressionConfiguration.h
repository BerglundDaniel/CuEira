#ifndef CUDALOGISTICREGRESSIONCONFIGURATION_H_
#define CUDALOGISTICREGRESSIONCONFIGURATION_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <Configuration.h>
#include <CudaAdapter.cu>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <CudaException.h>
#include <CublasException.h>
#include <DimensionMismatch.h>
#include <HostToDevice.h>
#include <DeviceToHost.h>
#include <PinnedHostVector.h>
#include <PinnedHostMatrix.h>
#include <RegularHostVector.h>
#include <RegularHostMatrix.h>
#include <KernelWrapper.h>
#include <MKLWrapper.h>
#include <LogisticRegressionConfiguration.h>

#ifdef PROFILE
#include <Event.h>
#endif

namespace CuEira {
namespace Model {
namespace LogisticRegression {
namespace CUDA {

using namespace CuEira::CUDA;
using namespace CuEira::Container;

/**
 * This is class is responsible for making the needed transfers of the data to the GPU and putting everything at the correct place
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaLogisticRegressionConfiguration: public LogisticRegressionConfiguration {
public:

  /**
   * This
   */
  CudaLogisticRegressionConfiguration(const Configuration& configuration, const HostToDevice& hostToDevice,
      const DeviceToHost& deviceToHost, const DeviceVector& deviceOutcomes, const KernelWrapper& kernelWrapper,
      const MKLWrapper& blasWrapper);

  /**
   * This
   */
  CudaLogisticRegressionConfiguration(const Configuration& configuration, const HostToDevice& hostToDevice,
      const DeviceToHost& deviceToHost, const DeviceVector& deviceOutcomes, const KernelWrapper& kernelWrapper,
      const MKLWrapper& blasWrapper, const PinnedHostMatrix& covariates);

  /**
   * Frees all related memory
   */
  virtual ~CudaLogisticRegressionConfiguration();

  virtual void setEnvironmentFactor(const HostVector& environmentData);
  virtual void setSNP(const HostVector& snpData);
  virtual void setInteraction(const HostVector& interactionVector);

  virtual const KernelWrapper& getKernelWrapper() const;
  virtual const HostToDevice& getHostToDevice() const;
  virtual const DeviceToHost& getDeviceToHost() const;

  virtual const DeviceMatrix& getPredictors() const;
  virtual const DeviceVector& getOutcomes() const;
  virtual DeviceVector& getProbabilites();
  virtual DeviceVector& getScores();
  virtual HostVector& getScoresHost();
  virtual DeviceMatrix& getInformationMatrix();
  virtual DeviceVector& getBetaCoefficents();

  virtual DeviceMatrix& getWorkMatrixNxM();
  virtual DeviceVector& getWorkVectorNx1();

  virtual const PinnedHostVector& getDefaultBetaCoefficents() const;

  CudaLogisticRegressionConfiguration(const CudaLogisticRegressionConfiguration&) = delete;
  CudaLogisticRegressionConfiguration(CudaLogisticRegressionConfiguration&&) = delete;
  CudaLogisticRegressionConfiguration& operator=(const CudaLogisticRegressionConfiguration&) = delete;
  CudaLogisticRegressionConfiguration& operator=(CudaLogisticRegressionConfiguration&&) = delete;

#ifdef PROFILE
  Event* beforeCov;
  Event* afterCov;

  Event* beforeIntercept;
  Event* afterIntercept;

  Event* beforeSNP;
  Event* afterSNP;

  Event* beforeEnv;
  Event* afterEnv;

  Event* beforeInter;
  Event* afterInter;
#endif

protected:
  CudaLogisticRegressionConfiguration(const Configuration& configuration, const MKLWrapper& blasWrapper); //For the mock

  void transferIntercept();

  const KernelWrapper* kernelWrapper;
  const HostToDevice* hostToDevice;
  const DeviceToHost* deviceToHost;

  DeviceMatrix* devicePredictors;
  const DeviceVector* deviceOutcomes;

  PRECISION* devicePredictorsMemoryPointer;
  PinnedHostVector* betaCoefficentsDefaultHost;
  PinnedHostVector* scoresHost;

  DeviceVector* betaCoefficentsDevice;
  DeviceVector* probabilitesDevice;
  DeviceVector* scoresDevice;
  DeviceMatrix* informationMatrixDevice;
  DeviceMatrix* workMatrixNxMDevice;
  DeviceVector* workVectorNx1Device;
};

} /* namespace CUDA */
} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif /* CUDALOGISTICREGRESSIONCONFIGURATION_H_ */
