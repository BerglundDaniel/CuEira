#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

#include <CudaAdapter.cu>
#include <CudaException.h>
#include <CublasException.h>
#include <KernelWrapper.h>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <PinnedHostMatrix.h>
#include <PinnedHostVector.h>
#include <DeviceToHost.h>
#include <HostToDevice.h>
#include <Device.h>
#include <Stream.h>
#include <StreamFactory.h>

using ::testing::Ge;
using ::testing::Le;

namespace CuEira {
namespace CUDA {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LogLikelihoodPartsTest: public ::testing::Test {
protected:
  LogLikelihoodPartsTest();
  virtual ~LogLikelihoodPartsTest();
  virtual void SetUp();
  virtual void TearDown();

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
  KernelWrapper kernelWrapper;
};

LogLikelihoodPartsTest::LogLikelihoodPartsTest() :
    device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDeviceStream1(*stream), deviceToHostStream1(
        *stream), kernelWrapper(*stream) {

}

LogLikelihoodPartsTest::~LogLikelihoodPartsTest() {
  delete stream;
}

void LogLikelihoodPartsTest::SetUp() {

}

void LogLikelihoodPartsTest::TearDown() {

}

TEST_F(LogLikelihoodPartsTest, KernelSmallVector) {
  double e = 10e-5;
  const int numberOfRows = 5;

  Container::PinnedHostVector* hostVectorOutcomes = new Container::PinnedHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    if(i < 4){
      (*hostVectorOutcomes)(i) = 0.9;
    }else{
      (*hostVectorOutcomes)(i) = 0.1;
    }

  }

  Container::PinnedHostVector* hostVectorProbabilites = new Container::PinnedHostVector(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    if(i < 2){
      (*hostVectorProbabilites)(i) = 0.3;
    }else if(i < 6){
      (*hostVectorProbabilites)(i) = 0.7;
    }else{
      (*hostVectorProbabilites)(i) = 0.4;
    }
  }

  Container::DeviceVector* outcomesDeviceVector = hostToDeviceStream1.transferVector(*hostVectorOutcomes);
  Container::DeviceVector* probabilitesDeviceVector = hostToDeviceStream1.transferVector(*hostVectorProbabilites);
  Container::DeviceVector* resultDeviceVector = new Container::DeviceVector(numberOfRows);

  kernelWrapper.logLikelihoodParts(*outcomesDeviceVector, *probabilitesDeviceVector, *resultDeviceVector);

  Container::HostVector* resultHostVector = deviceToHostStream1.transferVector(*resultDeviceVector);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error in ElemtWiseDivisionTest: ");

  ASSERT_EQ(numberOfRows, resultHostVector->getNumberOfRows());

  for(int i = 0; i < numberOfRows; ++i){
    PRECISION x = (*hostVectorOutcomes)(i) * log((*hostVectorProbabilites)(i))
        + (1 - (*hostVectorOutcomes)(i)) * log(1 - (*hostVectorProbabilites)(i));

    double l = x - e;
    double h = x + e;

    EXPECT_THAT((*resultHostVector)(i), Ge(l));
    EXPECT_THAT((*resultHostVector)(i), Le(h));
  }

  delete hostVectorOutcomes;
  delete hostVectorProbabilites;
  delete outcomesDeviceVector;
  delete probabilitesDeviceVector;
  delete resultDeviceVector;
  delete resultHostVector;
}

TEST_F(LogLikelihoodPartsTest, KernelException) {
  const int numberOfRows = 5;

  Container::DeviceVector* outcomesDeviceVector = new Container::DeviceVector(numberOfRows);
  Container::DeviceVector* probabilitesDeviceVector = new Container::DeviceVector(numberOfRows + 1);
  Container::DeviceVector* resultDeviceVector = new Container::DeviceVector(numberOfRows);

  EXPECT_THROW(kernelWrapper.logLikelihoodParts(*outcomesDeviceVector, *probabilitesDeviceVector, *resultDeviceVector),
      CudaException);

  delete outcomesDeviceVector;
  delete probabilitesDeviceVector;
  delete resultDeviceVector;

  outcomesDeviceVector = new Container::DeviceVector(numberOfRows + 1);
  probabilitesDeviceVector = new Container::DeviceVector(numberOfRows + 1);
  resultDeviceVector = new Container::DeviceVector(numberOfRows);
  EXPECT_THROW(kernelWrapper.logLikelihoodParts(*outcomesDeviceVector, *probabilitesDeviceVector, *resultDeviceVector),
      CudaException);

  delete outcomesDeviceVector;
  delete probabilitesDeviceVector;
  delete resultDeviceVector;

  outcomesDeviceVector = new Container::DeviceVector(numberOfRows + 1);
  probabilitesDeviceVector = new Container::DeviceVector(numberOfRows);
  resultDeviceVector = new Container::DeviceVector(numberOfRows + 1);
  EXPECT_THROW(kernelWrapper.logLikelihoodParts(*outcomesDeviceVector, *probabilitesDeviceVector, *resultDeviceVector),
      CudaException);

  delete outcomesDeviceVector;
  delete probabilitesDeviceVector;
  delete resultDeviceVector;

  outcomesDeviceVector = new Container::DeviceVector(numberOfRows);
  probabilitesDeviceVector = new Container::DeviceVector(numberOfRows - 1);
  resultDeviceVector = new Container::DeviceVector(numberOfRows + 1);
  EXPECT_THROW(kernelWrapper.logLikelihoodParts(*outcomesDeviceVector, *probabilitesDeviceVector, *resultDeviceVector),
      CudaException);

  delete outcomesDeviceVector;
  delete probabilitesDeviceVector;
  delete resultDeviceVector;

  outcomesDeviceVector = new Container::DeviceVector(numberOfRows + 1);
  probabilitesDeviceVector = new Container::DeviceVector(numberOfRows);
  resultDeviceVector = new Container::DeviceVector(numberOfRows - 1);
  EXPECT_THROW(kernelWrapper.logLikelihoodParts(*outcomesDeviceVector, *probabilitesDeviceVector, *resultDeviceVector),
      CudaException);

  delete outcomesDeviceVector;
  delete probabilitesDeviceVector;
  delete resultDeviceVector;
}

} /* namespace CUDA */
} /* namespace CuEira */

