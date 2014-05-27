#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>

#include <BedReader.h>
#include <BimReader.h>
#include <FamReader.h>
#include <PlinkReader.h>
#include <PlinkReaderFactory.h>
#include <DataFilesReader.h>
#include <DataFilesReaderFactory.h>
#include <ConfigurationMock.h>
#include <GeneticModel.h>
#include <Person.h>
#include <Sex.h>
#include <Id.h>
#include <Phenotype.h>
#include <SNP.h>
#include <ConstructorHelpers.h>
#include <HostMatrix.h>
#include <HostVector.h>
#include <FileReaderException.h>

using testing::Return;
using testing::_;
using testing::ReturnRef;
using testing::Eq;
using testing::AtLeast;

namespace CuEira {
namespace FileIO {

/**
 * Test for testing ....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class FileIOIntegrationTest: public ::testing::Test {
protected:
  FileIOIntegrationTest();
  virtual ~FileIOIntegrationTest();
  virtual void SetUp();
  virtual void TearDown();

  static const int numberOfIndividualsTotalStatic = 10;
  static const int numberOfIndividualsToIncludeStatic = 6;
  const int numberOfIndividualsTotal;
  const int numberOfIndividualsToInclude;
  ConfigurationMock configMock;
};

FileIOIntegrationTest::FileIOIntegrationTest() :
    numberOfIndividualsTotal(numberOfIndividualsTotalStatic), numberOfIndividualsToInclude(
        numberOfIndividualsToIncludeStatic) {

}

FileIOIntegrationTest::~FileIOIntegrationTest() {

}

void FileIOIntegrationTest::SetUp() {
  //Expect Configuration
  EXPECT_CALL(configMock, getMinorAlleleFrequencyThreshold()).Times(AtLeast(0)).WillRepeatedly(Return(0.05));
  EXPECT_CALL(configMock, getBedFilePath()).Times(AtLeast(1)).WillRepeatedly(Return(std::string(CuEira_BUILD_DIR)+std::string("/test.bed")));
  EXPECT_CALL(configMock, getBimFilePath()).Times(AtLeast(1)).WillRepeatedly(Return(std::string(CuEira_BUILD_DIR)+std::string("/test.bim")));
  EXPECT_CALL(configMock, getFamFilePath()).Times(AtLeast(1)).WillRepeatedly(Return(std::string(CuEira_BUILD_DIR)+std::string("/test.fam")));
  EXPECT_CALL(configMock, getEnvironmentFilePath()).Times(AtLeast(1)).WillRepeatedly(Return(std::string(CuEira_BUILD_DIR)+std::string("/test_env.txt")));
  EXPECT_CALL(configMock, getCovariateFilePath()).Times(AtLeast(1)).WillRepeatedly(Return(std::string(CuEira_BUILD_DIR)+std::string("/test_cov.txt")));

  EXPECT_CALL(configMock, getEnvironmentIndividualIdColumnName()).Times(AtLeast(1)).WillRepeatedly(Return("indid"));
  EXPECT_CALL(configMock, getCovariateIndividualIdColumnName()).Times(AtLeast(1)).WillRepeatedly(Return("indid"));
  EXPECT_CALL(configMock, getGeneticModel()).Times(AtLeast(1)).WillRepeatedly(Return(DOMINANT));

  EXPECT_CALL(configMock, getEnvironmentDelimiter()).Times(AtLeast(1)).WillRepeatedly(Return("\t "));
  EXPECT_CALL(configMock, getCovariateDelimiter()).Times(AtLeast(1)).WillRepeatedly(Return("\t "));

  EXPECT_CALL(configMock, excludeSNPsWithNegativePosition()).Times(AtLeast(1)).WillRepeatedly(Return(true));
  EXPECT_CALL(configMock, getPhenotypeCoding()).Times(AtLeast(1)).WillRepeatedly(Return(ONE_TWO_CODING));

}

void FileIOIntegrationTest::TearDown() {

}

TEST_F(FileIOIntegrationTest, ConstructAndBasicGetters) {
  PlinkReaderFactory plinkReaderFactory;
  DataFilesReaderFactory dataFilesReaderFactory(plinkReaderFactory);
  DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configMock);

  EXPECT_EQ(2, dataFilesReader->getNumberOfCovariates());
  EXPECT_EQ(2, dataFilesReader->getNumberOfEnvironmentFactors());

  EXPECT_EQ(10, dataFilesReader->getPersonHandler().getNumberOfIndividualsTotal());

  delete dataFilesReader;
}

TEST_F(FileIOIntegrationTest, GetSNPInfo) {
  PlinkReaderFactory plinkReaderFactory;
  DataFilesReaderFactory dataFilesReaderFactory(plinkReaderFactory);
  DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configMock);

  std::vector<SNP*> snpInformation = dataFilesReader->getSNPInformation();

  int numSNPToInclude = 0;
  int snpSize = snpInformation.size();
  for(int i = 0; i < snpSize; ++i){
    SNP* snp = snpInformation[i];
    if(snp->getInclude()){
      ++numSNPToInclude;
    }
  }

  EXPECT_EQ(10, snpSize);
  EXPECT_EQ(8, numSNPToInclude);

  delete dataFilesReader;
}

TEST_F(FileIOIntegrationTest, GetCovariates) {
  PlinkReaderFactory plinkReaderFactory;
  DataFilesReaderFactory dataFilesReaderFactory(plinkReaderFactory);
  DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configMock);

  const Container::HostMatrix& covariates = dataFilesReader->getCovariates();
  int covNumberOfRows = covariates.getNumberOfRows();
  ASSERT_EQ(10, covNumberOfRows);
  ASSERT_EQ(2, covariates.getNumberOfColumns());

  EXPECT_EQ(1, covariates(0, 0));
  EXPECT_EQ(2, covariates(1, 0));
  EXPECT_EQ(0, covariates(2, 0));
  EXPECT_EQ(0, covariates(3, 0));
  EXPECT_EQ(1, covariates(4, 0));
  EXPECT_EQ(1, covariates(5, 0));
  EXPECT_EQ(0, covariates(6, 0));
  EXPECT_EQ(1, covariates(7, 0));
  EXPECT_EQ(1, covariates(8, 0));
  EXPECT_EQ(0, covariates(9, 0));

  EXPECT_EQ(1, covariates(0, 1));
  EXPECT_EQ(0, covariates(1, 1));
  EXPECT_EQ(1, covariates(2, 1));
  EXPECT_EQ(1, covariates(3, 1));
  EXPECT_EQ(2, covariates(4, 1));
  EXPECT_EQ(0, covariates(5, 1));
  EXPECT_EQ(0, covariates(6, 1));
  EXPECT_EQ(1, covariates(7, 1));
  EXPECT_EQ(0, covariates(8, 1));
  EXPECT_EQ(2, covariates(9, 1));

  delete dataFilesReader;
}

TEST_F(FileIOIntegrationTest, GetEnvironment) {
  PlinkReaderFactory plinkReaderFactory;
  DataFilesReaderFactory dataFilesReaderFactory(plinkReaderFactory);
  DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configMock);

  Id id("env1");
  EnvironmentFactor envFactor(id);
  const Container::HostVector& env = dataFilesReader->getEnvironmentFactor(envFactor);
  int covNumberOfRows = env.getNumberOfRows();
  ASSERT_EQ(10, covNumberOfRows);
  ASSERT_EQ(1, env.getNumberOfColumns());

  EXPECT_EQ(1, env(0));
  EXPECT_EQ(1, env(1));
  EXPECT_EQ(0, env(2));
  EXPECT_EQ(0, env(3));
  EXPECT_EQ(1, env(4));
  EXPECT_EQ(1, env(5));
  EXPECT_EQ(0, env(6));
  EXPECT_EQ(1, env(7));
  EXPECT_EQ(1, env(8));
  EXPECT_EQ(0, env(9));

  delete dataFilesReader;
}

TEST_F(FileIOIntegrationTest, ReadSNP) {
  PlinkReaderFactory plinkReaderFactory;
  DataFilesReaderFactory dataFilesReaderFactory(plinkReaderFactory);
  DataFilesReader* dataFilesReader = dataFilesReaderFactory.constructDataFilesReader(configMock);

  Id id("SNP1");
  unsigned int pos = 3;
  std::string alleOneString("a1_1");
  std::string alleTwoString("a1_2");
  SNP snp(id, alleOneString, alleTwoString, pos);

  Container::SNPVector* snpVector = dataFilesReader->readSNP(snp);

  ASSERT_TRUE(snp.getInclude());
  const std::vector<int>* snpData = snpVector->getOrginalData();
  ASSERT_EQ(10, snpData->size());

  //Check maf and all freqs
  EXPECT_EQ(ALLELE_ONE, snp.getRiskAllele());
  EXPECT_EQ(snp.getAlleleOneAllFrequency(), snp.getMinorAlleleFrequency());

  EXPECT_EQ(6 / 10.0, snp.getAlleleOneCaseFrequency());
  EXPECT_EQ(4 / 10.0, snp.getAlleleTwoCaseFrequency());
  EXPECT_EQ(4 / 10.0, snp.getAlleleOneControlFrequency());
  EXPECT_EQ(6 / 10.0, snp.getAlleleTwoControlFrequency());
  EXPECT_EQ(0.5, snp.getAlleleOneAllFrequency());
  EXPECT_EQ(0.5, snp.getAlleleTwoAllFrequency());

  //Check data
  EXPECT_EQ(0, (*snpData)[0]);
  EXPECT_EQ(1, (*snpData)[1]);
  EXPECT_EQ(2, (*snpData)[2]);
  EXPECT_EQ(1, (*snpData)[3]);
  EXPECT_EQ(1, (*snpData)[4]);
  EXPECT_EQ(0, (*snpData)[5]);
  EXPECT_EQ(2, (*snpData)[6]);
  EXPECT_EQ(0, (*snpData)[7]);
  EXPECT_EQ(1, (*snpData)[8]);
  EXPECT_EQ(2, (*snpData)[9]);

  delete dataFilesReader;
}

}
/* namespace FileIO */
} /* namespace CuEira */

