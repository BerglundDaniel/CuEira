#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <ostream>

#include <CudaMultiplicativeInteractionModel.h>
#include <EnvironmentVectorMock.h>
#include <InteractionVectorMock.h>
#include <SNPVectorMock.h>
#include <PinnedHostMatrix.h>
#include <PinnedHostVector.h>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <DeviceToHost.h>
#include <HostToDevice.h>
#include <Device.h>
#include <Stream.h>
#include <StreamFactory.h>

using testing::Return;
using testing::ReturnRef;
using testing::AtLeast;

namespace CuEira {
namespace CUDA {

/**
 * Test for testing CudaMultiplicativeInteractionModel
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaMultiplicativeInteractionModelTest: public ::testing::Test {
protected:
  CudaMultiplicativeInteractionModelTest();
  virtual ~CudaMultiplicativeInteractionModelTest();
  virtual void SetUp();
  virtual void TearDown();

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
  KernelWrapper kernelWrapper;
};

CudaMultiplicativeInteractionModelTest::CudaMultiplicativeInteractionModelTest() :
    device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDeviceStream1(*stream), deviceToHostStream1(
        *stream), kernelWrapper(*stream) {

}

CudaMultiplicativeInteractionModelTest::~CudaMultiplicativeInteractionModelTest() {
  delete stream;
}

void CudaMultiplicativeInteractionModelTest::SetUp() {

}

void CudaMultiplicativeInteractionModelTest::TearDown() {

}

TEST_F(CudaMultiplicativeInteractionModelTest, Construct) {
  const int numberOfIndividuals = 10;
  MKLWrapper& mklWrapper;
  CudaMultiplicativeInteractionModel cudaMultiplicativeInteractionModel(kernelWrapper);

  Container::PinnedHostVector snpData(numberOfIndividuals);
  Container::PinnedHostVector envData(numberOfIndividuals);
  Container::PinnedHostVector interData(numberOfIndividuals);

  for(int i = 0; i < numberOfIndividuals; ++i){
    snpData(i) = i;
  }

  for(int i = 0; i < numberOfIndividuals; ++i){
    envData(i) = numberOfIndividuals - i;
  }

  Container::DeviceVector* snpDataDevice = hostToDeviceStream1.transferVector(snpData);
  Container::DeviceVector* envDataDevice = hostToDeviceStream1.transferVector(envData);
  Container::DeviceVector* interDataDevice = hostToDeviceStream1.transferVector(interData);

  Container::SNPVectorMock<Container::DeviceVector> snpVectorMock;
  EXPECT_CALL(snpVectorMock, getNumberOfIndividualsToInclude()).WillRepeatedly(Return(numberOfIndividuals));
  EXPECT_CALL(snpVectorMock, getOriginalSNPData()).Times(1).WillRepeatedly(ReturnRef(*snpDataDevice));

  Container::EnvironmentVectorMock<Container::DeviceVector> environmentVectorMock;
  EXPECT_CALL(environmentVectorMock, getNumberOfIndividualsToInclude()).WillRepeatedly(Return(numberOfIndividuals));
  EXPECT_CALL(environmentVectorMock, getEnvironmentData()).Times(1).WillRepeatedly(ReturnRef(*envDataDevice));

  Container::InteractionVectorMock<Container::DeviceVector> interactionVectorMock;
  EXPECT_CALL(interactionVectorMock, getNumberOfIndividualsToInclude()).WillRepeatedly(Return(numberOfIndividuals));
  EXPECT_CALL(interactionVectorMock, getInteractionData()).Times(1).WillRepeatedly(ReturnRef(*interDataDevice));
  EXPECT_CALL(interactionVectorMock, updateSize(numberOfIndividuals)).Times(1);

  cudaMultiplicativeInteractionModel.applyModel(snpVectorMock, environmentVectorMock, interactionVectorMock);

  Container::PinnedHostVector* interDataHostRes = deviceToHostStream1.transferVector(*interDataDevice);

  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ(i * (numberOfIndividuals - i), (*interDataHostRes)(i));
  }

  delete snpDataDevice;
  delete envDataDevice;
  delete interDataDevice;
  delete interDataHostRes;
}

} /* namespace CUDA */
} /* namespace CuEira */
