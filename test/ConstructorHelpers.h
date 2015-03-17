#ifndef CONSTRUCTORHELPERS_H_
#define CONSTRUCTORHELPERS_H_

#include <sstream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <Id.h>
#include <Sex.h>
#include <Person.h>
#include <Phenotype.h>
#include <SNPVectorMock.h>
#include <EnvironmentVectorMock.h>
#include <EnvironmentFactor.h>
#include <EnvironmentFactorHandler.h>
#include <EnvironmentFactorHandlerMock.h>
#include <SNP.h>
#include <GeneticModel.h>
#include <BedReaderMock.h>
#include <Configuration.h>
#include <PersonHandlerMock.h>
#include <ConfigurationMock.h>
#include <SNPVectorFactoryMock.h>
#include <ContingencyTableFactoryMock.h>
#include <AlleleStatisticsFactoryMock.h>
#include <StreamMock.h>
#include <DeviceMock.h>
#include <CudaAdapter.cu>

#ifdef CPU
#include <lapackpp/gmd.h>
#include <LapackppHostMatrix.h>
#include <lapackpp/lavd.h>
#else
#include <PinnedHostMatrix.h>
#include <PinnedHostVector.h>
#endif

using testing::Return;

namespace CuEira {
namespace CuEira_Test {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ConstructorHelpers {
public:
  ConstructorHelpers();
  virtual ~ConstructorHelpers();

  Container::EnvironmentVectorMock* constructEnvironmentVectorMock();
  EnvironmentFactorHandlerMock* constructEnvironmentFactorHandlerMock();
  FileIO::BedReaderMock* constructBedReaderMock();
  ContingencyTableFactoryMock* constructContingencyTableFactoryMock();
  CUDA::StreamMock* constructStreamMock();

private:

};

} /* namespace CuEira_Test */
} /* namespace CuEira */

#endif /* CONSTRUCTORHELPERS_H_ */
