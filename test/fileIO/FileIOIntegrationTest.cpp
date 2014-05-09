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
};

FileIOIntegrationTest::FileIOIntegrationTest() {

}

FileIOIntegrationTest::~FileIOIntegrationTest() {

}

void FileIOIntegrationTest::SetUp() {

}

void FileIOIntegrationTest::TearDown() {

}

TEST_F(FileIOIntegrationTest, Construct) {

}

}
/* namespace FileIO */
} /* namespace CuEira */

