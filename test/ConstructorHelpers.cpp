#include <ConstructorHelpers.h>

namespace CuEira {
namespace CuEira_Test {

ConstructorHelpers::ConstructorHelpers() {
  srand(time(NULL));
}

ConstructorHelpers::~ConstructorHelpers() {

}

Container::EnvironmentVectorMock* ConstructorHelpers::constructEnvironmentVectorMock() {
  return new Container::EnvironmentVectorMock();
}

EnvironmentFactorHandlerMock* ConstructorHelpers::constructEnvironmentFactorHandlerMock() {
  const int numberOfIndividuals = 3;
  const int numberOfColumns = 2;

#ifdef CPU
  Container::HostMatrix* dataMatrix= new Container::LapackppHostMatrix(
      new LaGenMatDouble(numberOfIndividuals, numberOfColumns));
#else
  Container::HostMatrix* dataMatrix = new Container::PinnedHostMatrix(numberOfIndividuals, numberOfColumns);
#endif

  std::vector<EnvironmentFactor*>* environmentFactors = new std::vector<EnvironmentFactor*>(numberOfColumns);
  for(int i = 0; i < numberOfColumns; ++i){
    std::ostringstream os;
    os << "envfactor" << i;
    Id id(os.str());
    (*environmentFactors)[i] = new EnvironmentFactor(id);
  }

  return new EnvironmentFactorHandlerMock(dataMatrix, environmentFactors);
}

FileIO::BedReaderMock* ConstructorHelpers::constructBedReaderMock() {
  ConfigurationMock configurationMock;
  EXPECT_CALL(configurationMock, getGeneticModel()).WillRepeatedly(Return(DOMINANT));

  return new FileIO::BedReaderMock(configurationMock, Container::SNPVectorFactoryMock(configurationMock),
      AlleleStatisticsFactoryMock(), PersonHandlerMock());
}

ContingencyTableFactoryMock* ConstructorHelpers::constructContingencyTableFactoryMock() {
  const int size = 3;
#ifdef CPU
  Container::LapackppHostVector outcomes(new LaVectorDouble(size));
#else
  Container::PinnedHostVector outcomes(size);
#endif
  return new ContingencyTableFactoryMock(outcomes);
}

CUDA::StreamMock* ConstructorHelpers::constructStreamMock() {
  cudaStream_t* cudaStream = new cudaStream_t();
  cublasHandle_t* cublasHandle = new cublasHandle_t();

  CUDA::handleCublasStatus(cublasCreate(cublasHandle), "Failed to create new cublas handle:");
  CUDA::handleCudaStatus(cudaStreamCreate(cudaStream), "Failed to create new cuda stream:");
  CUDA::handleCublasStatus(cublasSetStream(*cublasHandle, *cudaStream), "Failed to set cuda stream for cublas handle:");

  return new CUDA::StreamMock(CUDA::DeviceMock(), cudaStream, cublasHandle);
}

}
/* namespace CuEira_Test */
} /* namespace CuEira */
