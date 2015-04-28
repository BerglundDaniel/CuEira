#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <Recode.h>
#include <CudaEnvironmentVector.h>
#include <EnvironmentFactor.h>
#include <VariableType.h>
#include <Id.h>
#include <EnvironmentFactorHandlerMock.h>
#include <MissingDataHandlerMock.h>
#include <InvalidState.h>
#include <DeviceVector.h>
#include <PinnedHostVector.h>
#include <DeviceToHost.h>
#include <HostToDevice.h>
#include <Device.h>
#include <Stream.h>
#include <StreamFactory.h>
#include <KernelWrapper.h>
#include <CublasWrapper.h>

using testing::Return;
using testing::AtLeast;
using testing::ReturnRef;
using testing::Eq;
using testing::ByRef;
using testing::Invoke;

namespace CuEira {
namespace Container {
namespace CUDA {

using namespace CuEira::CUDA;

/**
 * Test for testing CudaEnvironmentVector
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CudaEnvironmentVectorTest: public ::testing::Test {
protected:
  CudaEnvironmentVectorTest();
  virtual ~CudaEnvironmentVectorTest();
  virtual void SetUp();
  virtual void TearDown();

  void FillExMissingVector(const Container::DeviceVector& fromVector, Container::DeviceVector& toVector);

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDevice;
  DeviceToHost deviceToHost;
  KernelWrapper kernelWrapper;
  CublasWrapper cublasWrapper;

  const int numberOfIndividualsTotal;
  EnvironmentFactorHandlerMock<DeviceVector> environmentFactorHandlerMock;
  EnvironmentFactor envFactor;
  EnvironmentFactor binaryEnvFactor;
  PinnedHostVector normalData;
  PinnedHostVector binaryData;
  DeviceVector* normalDataDevice;
  DeviceVector* binaryDataDevice;
};

void CudaEnvironmentVectorTest::FillExMissingVector(const Container::DeviceVector& fromVector,
    Container::DeviceVector& toVector) {
  const int orgFromSize = fromVector.getNumberOfRows();
  const int toSize = toVector.getNumberOfRows();
  fromVector.updateSize(toSize);

  cublasWrapper.copyVector(fromVector, toVector);
}

CudaEnvironmentVectorTest::CudaEnvironmentVectorTest() :
    device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDevice(*stream), deviceToHost(
        *stream), numberOfIndividualsTotal(9), envFactor(Id("startFactor")), binaryEnvFactor(Id("binaryFactor")), normalData(
        numberOfIndividualsTotal), binaryData(numberOfIndividualsTotal), kernelWrapper(*stream), cublasWrapper(*stream), normalDataDevice(
        nullptr), binaryDataDevice(nullptr) {
  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    normalData(i) = i;

    if(i < 5 && i > 2){
      binaryData(i) = 1;
    }else{
      binaryData(i) = 0;
    }
  }

  envFactor.setVariableType(OTHER);
  binaryEnvFactor.setVariableType(BINARY);

  envFactor.setMax(numberOfIndividualsTotal - 1);
  envFactor.setMin(-1);
  normalData(3) = -1;

  binaryEnvFactor.setMax(1);
  binaryEnvFactor.setMin(0);

  normalDataDevice = hostToDevice.transferVector(normalData);
  binaryDataDevice = hostToDevice.transferVector(binaryData);
}

CudaEnvironmentVectorTest::~CudaEnvironmentVectorTest() {
  delete normalDataDevice;
  delete binaryDataDevice;
}

void CudaEnvironmentVectorTest::SetUp() {
  EXPECT_CALL(environmentFactorHandlerMock, getNumberOfIndividualsTotal()).Times(1).WillRepeatedly(
      Return(numberOfIndividualsTotal));
}

void CudaEnvironmentVectorTest::TearDown() {

}

#ifdef DEBUG
TEST_F(CudaEnvironmentVectorTest, ConstructAndGetException){
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(envFactor));
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentData()).Times(1).WillRepeatedly(ReturnRef(*normalDataDevice));

  CudaEnvironmentVector cudaEnvironmentVector(environmentFactorHandlerMock, kernelWrapper, cublasWrapper);

  EXPECT_THROW(cudaEnvironmentVector.getNumberOfIndividualsToInclude(), InvalidState);
  EXPECT_THROW(cudaEnvironmentVector.getEnvironmentData(), InvalidState);
}
#endif

TEST_F(CudaEnvironmentVectorTest, RecodeBinaryNoMissing) {
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(binaryEnvFactor));
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentData()).Times(1).WillRepeatedly(ReturnRef(*binaryDataDevice));

  CudaEnvironmentVector cudaEnvironmentVector(environmentFactorHandlerMock, kernelWrapper, cublasWrapper);
  EXPECT_EQ(binaryEnvFactor, cudaEnvironmentVector.getEnvironmentFactor());
  EXPECT_EQ(numberOfIndividualsTotal, cudaEnvironmentVector.getNumberOfIndividualsTotal());

  //ALL_RISK
  cudaEnvironmentVector.recode(ALL_RISK);
  EXPECT_EQ(numberOfIndividualsTotal, cudaEnvironmentVector.getNumberOfIndividualsToInclude());
  const PinnedHostVector& allRiskVectorDevice1 = cudaEnvironmentVector.getEnvironmentData();
  const PinnedHostVector* allRiskVector1 = deviceToHost.transferVector(allRiskVectorDevice1);

  EXPECT_EQ(numberOfIndividualsTotal, allRiskVector1->getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(binaryData(i), (*allRiskVector1)(i));
  }

  //ENVIRONMENT_PROTECT
  cudaEnvironmentVector.recode(ENVIRONMENT_PROTECT);
  EXPECT_EQ(numberOfIndividualsTotal, cudaEnvironmentVector.getNumberOfIndividualsToInclude());
  const DeviceVector& envProctVectorDevice1 = cudaEnvironmentVector.getEnvironmentData();
  const PinnedHostVector* envProctVector1 = deviceToHost.transferVector(envProctVectorDevice1);
  EXPECT_EQ(numberOfIndividualsTotal, envProctVector1->getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(1 - binaryData(i), (*envProctVector1)(i));
  }

  //INTERACTION_PROTECT
  cudaEnvironmentVector.recode(INTERACTION_PROTECT);
  EXPECT_EQ(numberOfIndividualsTotal, cudaEnvironmentVector.getNumberOfIndividualsToInclude());
  const DeviceVector& interProctVectorDevice1 = cudaEnvironmentVector.getEnvironmentData();
  const PinnedHostVector* interProctVector1 = deviceToHost.transferVector(interProctVectorDevice1);
  EXPECT_EQ(numberOfIndividualsTotal, interProctVector1->getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(1 - binaryData(i), (*interProctVector1)(i));
  }

  //ALL_RISK
  cudaEnvironmentVector.recode(ALL_RISK);
  EXPECT_EQ(numberOfIndividualsTotal, cudaEnvironmentVector.getNumberOfIndividualsToInclude());
  const DeviceVector& allRiskVectorDevice2 = cudaEnvironmentVector.getEnvironmentData();
  const PinnedHostVector* allRiskVector2 = deviceToHost.transferVector(allRiskVectorDevice2);
  EXPECT_EQ(numberOfIndividualsTotal, allRiskVector2->getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(binaryData(i), (*allRiskVector2)(i));
  }

  delete allRiskVector1;
  delete envProctVector1;
  delete interProctVector1;
  delete allRiskVector2;
}

TEST_F(CudaEnvironmentVectorTest, RecodeBinaryMissing) {
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(binaryEnvFactor));
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentData()).Times(1).WillRepeatedly(ReturnRef(*binaryDataDevice));

  CudaEnvironmentVector cudaEnvironmentVector(environmentFactorHandlerMock, kernelWrapper, cublasWrapper);
  EXPECT_EQ(binaryEnvFactor, cudaEnvironmentVector.getEnvironmentFactor());
  EXPECT_EQ(numberOfIndividualsTotal, cudaEnvironmentVector.getNumberOfIndividualsTotal());

  MissingDataHandlerMock<PinnedHostVector> missingDataHandlerMock;

  EXPECT_CALL(missingDataHandlerMock, copyNonMissing(_,_)).Times(2).WillRepeatedly(Invoke(this, &FillExMissingVector));

  //INTERACTION_PROTECT
  const int numberOfIndvidualsExMissing1 = numberOfIndividualsTotal - 4;
  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillOnce(
      Return(numberOfIndvidualsExMissing1));

  cudaEnvironmentVector.recode(INTERACTION_PROTECT, missingDataHandlerMock);

  const DeviceVector& interProctVectorDevice1 = cudaEnvironmentVector.getEnvironmentData();
  const PinnedHostVector* interProctVector1 = deviceToHost.transferVector(interProctVectorDevice1);
  EXPECT_EQ(numberOfIndividualsTotal, interProctVector1->getNumberOfRows());

  for(int i = 0; i < numberOfIndvidualsExMissing1; ++i){
    EXPECT_EQ(1 - binaryData(i), (*interProctVector1)(i));
  }

  //ALL_RISK
  const int numberOfIndvidualsExMissing2 = numberOfIndividualsTotal - 3;
  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillOnce(
      Return(numberOfIndvidualsExMissing2));

  cudaEnvironmentVector.recode(ALL_RISK, missingDataHandlerMock);

  const DeviceVector& allRiskVectorDevice1 = cudaEnvironmentVector.getEnvironmentData();
  const PinnedHostVector* allRiskVector1 = deviceToHost.transferVector(allRiskVectorDevice1);
  EXPECT_EQ(numberOfIndividualsTotal, allRiskVector1->getNumberOfRows());

  for(int i = 0; i < numberOfIndvidualsExMissing2; ++i){
    EXPECT_EQ(binaryData(i), (*allRiskVector1)(i));
  }

  delete allRiskVector1;
  delete interProctVector1;
}

TEST_F(CudaEnvironmentVectorTest, RecodeNonBinaryNoMissing) {
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(envFactor));
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentData()).Times(1).WillRepeatedly(ReturnRef(*normalDataDevice));

  const int c = envFactor.getMax() - envFactor.getMin();

  CudaEnvironmentVector cudaEnvironmentVector(environmentFactorHandlerMock, kernelWrapper, cublasWrapper);
  EXPECT_EQ(envFactor, cudaEnvironmentVector.getEnvironmentFactor());
  EXPECT_EQ(numberOfIndividualsTotal, cudaEnvironmentVector.getNumberOfIndividualsTotal());

  //ALL_RISK
  cudaEnvironmentVector.recode(ALL_RISK);
  EXPECT_EQ(numberOfIndividualsTotal, cudaEnvironmentVector.getNumberOfIndividualsToInclude());
  const DeviceVector& allRiskVectorDevice1 = cudaEnvironmentVector.getEnvironmentData();
  const PinnedHostVector* allRiskVector1 = deviceToHost.transferVector(allRiskVectorDevice1);
  EXPECT_EQ(numberOfIndividualsTotal, allRiskVector1->getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(normalData(i), (*allRiskVector1)(i));
  }

  //ENVIRONMENT_PROTECT
  cudaEnvironmentVector.recode(ENVIRONMENT_PROTECT);
  EXPECT_EQ(numberOfIndividualsTotal, cudaEnvironmentVector.getNumberOfIndividualsToInclude());
  const DeviceVector& envProctVectorDevice1 = cudaEnvironmentVector.getEnvironmentData();
  const PinnedHostVector* envProctVector1 = deviceToHost.transferVector(envProctVectorDevice1);
  EXPECT_EQ(numberOfIndividualsTotal, envProctVector1->getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(c - normalData(i), (*envProctVector1)(i));
  }

  //INTERACTION_PROTECT
  cudaEnvironmentVector.recode(INTERACTION_PROTECT);
  EXPECT_EQ(numberOfIndividualsTotal, cudaEnvironmentVector.getNumberOfIndividualsToInclude());
  const DeviceVector& interProctVectorDevice1 = cudaEnvironmentVector.getEnvironmentData();
  const PinnedHostVector* interProctVector1 = deviceToHost.transferVector(interProctVectorDevice1);
  EXPECT_EQ(numberOfIndividualsTotal, interProctVector1->getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(c - normalData(i), (*interProctVector1)(i));
  }

  //ALL_RISK
  cudaEnvironmentVector.recode(ALL_RISK);
  EXPECT_EQ(numberOfIndividualsTotal, cudaEnvironmentVector.getNumberOfIndividualsToInclude());
  const DeviceVector& allRiskVectorDevice2 = cudaEnvironmentVector.getEnvironmentData();
  const PinnedHostVector* allRiskVector2 = deviceToHost.transferVector(allRiskVectorDevice2);
  EXPECT_EQ(numberOfIndividualsTotal, allRiskVector2->getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(normalData(i), (*allRiskVector2)(i));
  }

  delete allRiskVector1;
  delete allRiskVector2;
  delete interProctVector1;
  delete envProctVector1;
}

TEST_F(CudaEnvironmentVectorTest, RecodeNonBinaryMissing) {
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(envFactor));
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentData()).Times(1).WillRepeatedly(ReturnRef(*normalDataDevice));

  const int c = envFactor.getMax() - envFactor.getMin();

  CudaEnvironmentVector cudaEnvironmentVector(environmentFactorHandlerMock, kernelWrapper, cublasWrapper);
  EXPECT_EQ(envFactor, cudaEnvironmentVector.getEnvironmentFactor());
  EXPECT_EQ(numberOfIndividualsTotal, cudaEnvironmentVector.getNumberOfIndividualsTotal());

  MissingDataHandlerMock<PinnedHostVector> missingDataHandlerMock;

  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillRepeatedly(
      Return(numberOfIndvidualsExMissing));
  EXPECT_CALL(missingDataHandlerMock, copyNonMissing(_,_)).Times(1);

  EXPECT_CALL(missingDataHandlerMock, copyNonMissing(_,_)).Times(2).WillRepeatedly(Invoke(this, &FillExMissingVector));

  //INTERACTION_PROTECT
  const int numberOfIndvidualsExMissing1 = numberOfIndividualsTotal - 4;
  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillOnce(
      Return(numberOfIndvidualsExMissing1));

  cudaEnvironmentVector.recode(INTERACTION_PROTECT, missingDataHandlerMock);

  const DeviceVector& interProctVectorDevice1 = cudaEnvironmentVector.getEnvironmentData();
  const PinnedHostVector* interProctVector1 = deviceToHost.transferVector(interProctVectorDevice1);
  EXPECT_EQ(numberOfIndividualsTotal, interProctVector1->getNumberOfRows());

  for(int i = 0; i < numberOfIndvidualsExMissing1; ++i){
    EXPECT_EQ(c - normalData(i), (*interProctVector1)(i));
  }

  //ALL_RISK
  const int numberOfIndvidualsExMissing2 = numberOfIndividualsTotal - 3;
  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillOnce(
      Return(numberOfIndvidualsExMissing2));

  cudaEnvironmentVector.recode(ALL_RISK, missingDataHandlerMock);

  const DeviceVector& allRiskVectorDevice1 = cudaEnvironmentVector.getEnvironmentData();
  const PinnedHostVector* allRiskVector1 = deviceToHost.transferVector(allRiskVectorDevice1);
  EXPECT_EQ(numberOfIndividualsTotal, allRiskVector1->getNumberOfRows());

  for(int i = 0; i < numberOfIndvidualsExMissing2; ++i){
    EXPECT_EQ(normalData(i), (*allRiskVector1)(i));
  }

  delete allRiskVector1;
  delete interProctVector1;
}

TEST_F(CudaEnvironmentVectorTest, RecodeBinaryMixed) {
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentFactor()).Times(1).WillRepeatedly(ReturnRef(binaryEnvFactor));
  EXPECT_CALL(environmentFactorHandlerMock, getEnvironmentData()).Times(1).WillRepeatedly(ReturnRef(*binaryDataDevice));

  CudaEnvironmentVector cudaEnvironmentVector(environmentFactorHandlerMock, kernelWrapper, cublasWrapper);
  EXPECT_EQ(binaryEnvFactor, cudaEnvironmentVector.getEnvironmentFactor());
  EXPECT_EQ(numberOfIndividualsTotal, cudaEnvironmentVector.getNumberOfIndividualsTotal());

  MissingDataHandlerMock<PinnedHostVector> missingDataHandlerMock;

  EXPECT_CALL(missingDataHandlerMock, copyNonMissing(_,_)).Times(2).WillRepeatedly(Invoke(this, &FillExMissingVector));

  //INTERACTION_PROTECT
  const int numberOfIndvidualsExMissing1 = numberOfIndividualsTotal - 4;
  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillOnce(
      Return(numberOfIndvidualsExMissing1));

  cudaEnvironmentVector.recode(INTERACTION_PROTECT, missingDataHandlerMock);

  const DeviceVector& interProctVectorDevice1 = cudaEnvironmentVector.getEnvironmentData();
  const PinnedHostVector* interProctVector1 = deviceToHost.transferVector(interProctVectorDevice1);
  EXPECT_EQ(numberOfIndividualsTotal, interProctVector1->getNumberOfRows());

  for(int i = 0; i < numberOfIndvidualsExMissing1; ++i){
    EXPECT_EQ(1 - binaryData(i), (*interProctVector1)(i));
  }

  //ALL_RISK
  cudaEnvironmentVector.recode(ALL_RISK);

  const DeviceVector& allRiskVectorDevice1 = cudaEnvironmentVector.getEnvironmentData();
  const PinnedHostVector* allRiskVector1 = deviceToHost.transferVector(allRiskVectorDevice1);
  EXPECT_EQ(numberOfIndividualsTotal, allRiskVector1->getNumberOfRows());

  for(int i = 0; i < numberOfIndividualsTotal; ++i){
    EXPECT_EQ(binaryData(i), (*allRiskVector1)(i));
  }

  //ENVIRONMENT_PROTECT
  const int numberOfIndvidualsExMissing3 = numberOfIndividualsTotal - 1;
  EXPECT_CALL(missingDataHandlerMock, getNumberOfIndividualsToInclude()).Times(1).WillOnce(
      Return(numberOfIndvidualsExMissing3));

  cudaEnvironmentVector.recode(ENVIRONMENT_PROTECT, missingDataHandlerMock);

  const DeviceVector& envProctVectorDevice1 = cudaEnvironmentVector.getEnvironmentData();
  const PinnedHostVector* envProctVector1 = deviceToHost.transferVector(envProctVectorDevice1);
  EXPECT_EQ(numberOfIndividualsTotal, envProctVector1->getNumberOfRows());

  for(int i = 0; i < numberOfIndvidualsExMissing3; ++i){
    EXPECT_EQ(1 - binaryData(i), (*envProctVector1)(i));
  }

  delete allRiskVector1;
  delete interProctVector1;
  delete envProctVector1;
}

} /* namespace CUDA */
} /* namespace Container */
} /* namespace CuEira */
