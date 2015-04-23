#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <sstream>
#include <ostream>

#include <CudaAdditiveInteractionModel.h>
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
 * Test for testing CudaAdditiveInteractionModel
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaAdditiveInteractionModelTest: public ::testing::Test {
protected:
  CudaAdditiveInteractionModelTest();
  virtual ~CudaAdditiveInteractionModelTest();
  virtual void SetUp();
  virtual void TearDown();

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
  KernelWrapper kernelWrapper;
};

CudaAdditiveInteractionModelTest::CudaAdditiveInteractionModelTest() :
    device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDeviceStream1(*stream), deviceToHostStream1(
        *stream), kernelWrapper(*stream) {

}

CudaAdditiveInteractionModelTest::~CudaAdditiveInteractionModelTest() {
  delete stream;
}

void CudaAdditiveInteractionModelTest::SetUp() {

}

void CudaAdditiveInteractionModelTest::TearDown() {

}

TEST_F(CudaAdditiveInteractionModelTest, Construct) {
  const int numberOfIndividuals = 10;
  MKLWrapper& mklWrapper;
  CudaAdditiveInteractionModel cudaAdditiveInteractionModel(kernelWrapper);

  Container::PinnedHostVector snpData(numberOfIndividuals);
  Container::PinnedHostVector envData(numberOfIndividuals);
  Container::PinnedHostVector interData(numberOfIndividuals);

  for(int i = 0; i < numberOfIndividuals; ++i){
    snpData(i) = i % 2;
  }

  for(int i = 0; i < numberOfIndividuals; ++i){
    envData(i) = (i + 2) % 3;
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

  cudaAdditiveInteractionModel.applyModel(snpVectorMock, environmentVectorMock, interactionVectorMock);

  Container::PinnedHostVector* snpDataHostRes = deviceToHostStream1.transferVector(*snpDataDevice);
  Container::PinnedHostVector* envDataHostRes = deviceToHostStream1.transferVector(*envDataDevice);
  Container::PinnedHostVector* interDataHostRes = deviceToHostStream1.transferVector(*interDataDevice);

  for(int i = 0; i < numberOfIndividuals; ++i){
    EXPECT_EQ(snpData(i) * envData(i), (*interDataHostRes)(i));

    if((*interDataHostRes)(i) == 0){
      EXPECT_EQ(snpData(i), (*snpDataHostRes)(i));
      EXPECT_EQ(envData(i), (*envDataHostRes)(i));
    }else{
      EXPECT_EQ(0, (*snpDataHostRes)(i));
      EXPECT_EQ(0, (*envDataHostRes)(i));
    }
  }

  delete snpDataDevice;
  delete envDataDevice;
  delete interDataDevice;

  delete snpDataHostRes;
  delete envDataHostRes;
  delete interDataHostRes;
}

} /* namespace CUDA */
} /* namespace CuEira */
