#ifndef LOGISTICREGRESSIONCONFIGURATIONMOCK_H_
#define LOGISTICREGRESSIONCONFIGURATIONMOCK_H_

#include <gmock/gmock.h>

#include <LogisticRegressionConfiguration.h>
#include <HostVector.h>
#include <KernelWrapper.h>
#include <DeviceMatrix.h>
#include <DeviceVector.h>

namespace CuEira {
namespace Model {
namespace LogisticRegression {

class LogisticRegressionConfigurationMock: public LogisticRegressionConfiguration {
public:
  LogisticRegressionConfigurationMock() :
      LogisticRegressionConfiguration() {

  }

  virtual ~LogisticRegressionConfigurationMock() {

  }

  MOCK_CONST_METHOD0(getNumberOfRows, int());
  MOCK_CONST_METHOD0(getNumberOfPredictors, int());
  MOCK_CONST_METHOD0(getNumberOfMaxIterations, int());
  MOCK_CONST_METHOD0(getConvergenceThreshold, double());
  MOCK_CONST_METHOD0(getKernelWrapper, const KernelWrapper&());
  MOCK_CONST_METHOD0(getPredictors, const Container::DeviceMatrix&());
  MOCK_CONST_METHOD0(getOutcomes, const Container::DeviceVector&());

  MOCK_METHOD0(getProbabilites, Container::DeviceVector&());
  MOCK_METHOD0(getScores, Container::DeviceVector&());
  MOCK_METHOD0(getInformationMatrix, Container::DeviceMatrix&());
  MOCK_METHOD0(getBetaCoefficents, Container::DeviceVector&());
  MOCK_METHOD0(getWorkMatrixNxM, Container::DeviceMatrix&());
  MOCK_METHOD0(getWorkVectorNx1, Container::DeviceVector&());

  MOCK_METHOD1(setEnvironmentFactor, void(const Container::HostVector&));
  MOCK_METHOD1(setSNP, void(const Container::HostVector&));
  MOCK_METHOD1(setInteraction, void(const Container::HostVector&));
};

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif /* LOGISTICREGRESSIONCONFIGURATIONMOCK_H_ */
