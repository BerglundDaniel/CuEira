#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

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
 * Test for testing the CalculateNumberOfAllelesPerGenotype kernel
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CalculateNumberOfAllelesPerGenotypeTest: public ::testing::Test {
protected:
  CalculateNumberOfAllelesPerGenotypeTest();
  virtual ~CalculateNumberOfAllelesPerGenotypeTest();
  virtual void SetUp();
  virtual void TearDown();

  Device device;
  StreamFactory streamFactory;
  Stream* stream;
  HostToDevice hostToDeviceStream1;
  DeviceToHost deviceToHostStream1;
  KernelWrapper kernelWrapper;
};

CalculateNumberOfAllelesPerGenotypeTest::CalculateNumberOfAllelesPerGenotypeTest() :
    device(0), streamFactory(), stream(streamFactory.constructStream(device)), hostToDeviceStream1(*stream), deviceToHostStream1(
        *stream), kernelWrapper(*stream) {

}

CalculateNumberOfAllelesPerGenotypeTest::~CalculateNumberOfAllelesPerGenotypeTest() {
  delete stream;
}

void CalculateNumberOfAllelesPerGenotypeTest::SetUp() {

}

void CalculateNumberOfAllelesPerGenotypeTest::TearDown() {

}

TEST_F(CalculateNumberOfAllelesPerGenotypeTest, KernelSmall) {
  const int numberOfRows = 10;

  Container::PinnedHostVector snpDataHost(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    snpDataHost(i) = i % 3;
  }

  Container::PinnedHostVector phenotypeDataHost(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    if(i < numberOfRows / 2){
      phenotypeDataHost(i) = 0;
    }else{
      phenotypeDataHost(i) = 1;
    }
  }

  Container::DeviceVector* snpDataDevice = hostToDeviceStream1.transferVector(snpDataHost);
  Container::DeviceVector* phenotypeDataDevice = hostToDeviceStream1.transferVector(phenotypeDataHost);

  const Container::DeviceMatrix* numberOfAllelesPerGenotypeBlockDevice =
      kernelWrapper.calculateNumberOfAllelesPerGenotype(*snpDataDevice, *phenotypeDataDevice);

  const Container::HostMatrix* numberOfAllelesPerGenotypeBlockHost = deviceToHostStream1.transferMatrix(
      *numberOfAllelesPerGenotypeBlockDevice);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error in CalculateNumberOfAllelesPerGenotypeTest: ");

  const int numberOfBlocks = numberOfAllelesPerGenotypeBlockHost->getNumberOfRows();
  const int numberOfColumns = numberOfAllelesPerGenotypeBlockHost->getNumberOfColumns();

  ASSERT_TRUE(numberOfBlocks > 0);
  ASSERT_EQ(6, numberOfColumns);

  std::vector<int> numberOfAllelesPerGenotype(6);

  for(int i = 0; i < numberOfBlocks; ++i){
    for(int j = 0; j < 6; ++j){
      numberOfAllelesPerGenotype[j] += numberOfAllelesPerGenotypeBlockHost(i, j);
    }
  }

  ASSERT_EQ(2, numberOfAllelesPerGenotype[0]);
  ASSERT_EQ(2, numberOfAllelesPerGenotype[1]);
  ASSERT_EQ(1, numberOfAllelesPerGenotype[2]);
  ASSERT_EQ(2, numberOfAllelesPerGenotype[3]);
  ASSERT_EQ(1, numberOfAllelesPerGenotype[4]);
  ASSERT_EQ(2, numberOfAllelesPerGenotype[5]);

  delete snpDataDevice;
  delete phenotypeDataDevice;
  delete numberOfAllelesPerGenotypeBlockDevice;
  delete numberOfAllelesPerGenotypeBlockHost;
}

TEST_F(CalculateNumberOfAllelesPerGenotypeTest, KernelLarge) {
  const int numberOfRows = 1000;
  srand(time(NULL));

  Container::PinnedHostVector snpDataHost(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    snpDataHost(i) = rand() % 3;
  }

  Container::PinnedHostVector phenotypeDataHost(numberOfRows);
  for(int i = 0; i < numberOfRows; ++i){
    phenotypeDataHost(i) = rand() % 2;
  }

  Container::DeviceVector* snpDataDevice = hostToDeviceStream1.transferVector(snpDataHost);
  Container::DeviceVector* phenotypeDataDevice = hostToDeviceStream1.transferVector(phenotypeDataHost);

  const Container::DeviceMatrix* numberOfAllelesPerGenotypeBlockDevice =
      kernelWrapper.calculateNumberOfAllelesPerGenotype(*snpDataDevice, *phenotypeDataDevice);

  const Container::HostMatrix* numberOfAllelesPerGenotypeBlockHost = deviceToHostStream1.transferMatrix(
      *numberOfAllelesPerGenotypeBlockDevice);
  stream->syncStream();
  handleCudaStatus(cudaGetLastError(), "Error in CalculateNumberOfAllelesPerGenotypeTest: ");

  const int numberOfBlocks = numberOfAllelesPerGenotypeBlockHost->getNumberOfRows();
  const int numberOfColumns = numberOfAllelesPerGenotypeBlockHost->getNumberOfColumns();

  ASSERT_TRUE(numberOfBlocks > 0);
  ASSERT_EQ(6, numberOfColumns);

  std::vector<int> numberOfAllelesPerGenotype(6);

  for(int i = 0; i < numberOfBlocks; ++i){
    for(int j = 0; j < 6; ++j){
      numberOfAllelesPerGenotype[j] += numberOfAllelesPerGenotypeBlockHost(i, j);
    }
  }

  std::vector<int> numberOfAllelesPerGenotypeCPU(6);
  for(int i = 0; i < numberOfRows; ++i){
    ++numberOfAllelesPerGenotypeCPU[0][snpDataHost(i) + 3 * phenotypeDataHost(i)];
  }

  ASSERT_EQ(numberOfAllelesPerGenotypeCPU[0], numberOfAllelesPerGenotype[0]);
  ASSERT_EQ(numberOfAllelesPerGenotypeCPU[1], numberOfAllelesPerGenotype[1]);
  ASSERT_EQ(numberOfAllelesPerGenotypeCPU[2], numberOfAllelesPerGenotype[2]);
  ASSERT_EQ(numberOfAllelesPerGenotypeCPU[3], numberOfAllelesPerGenotype[3]);
  ASSERT_EQ(numberOfAllelesPerGenotypeCPU[4], numberOfAllelesPerGenotype[4]);
  ASSERT_EQ(numberOfAllelesPerGenotypeCPU[5], numberOfAllelesPerGenotype[5]);

  delete snpDataDevice;
  delete phenotypeDataDevice;
  delete numberOfAllelesPerGenotypeBlockDevice;
  delete numberOfAllelesPerGenotypeBlockHost;
}

#ifdef DEBUG
TEST_F(CalculateNumberOfAllelesPerGenotypeTest, KernelException){
  const int numberOfRows = 5;

  Container::DeviceVector* deviceVector1 = new Container::DeviceVector(numberOfRows + 1);
  Container::DeviceVector* deviceVector2 = new Container::DeviceVector(numberOfRows);

  EXPECT_THROW(kernelWrapper.calculateNumberOfAllelesPerGenotype(*deviceVector1, *deviceVector2),
      CudaException);

  delete deviceVector1;
  delete deviceVector2;
}
#endif

}
/* namespace CUDA */
} /* namespace CuEira */
