#ifndef FAMREADER_H_
#define FAMREADER_H_

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <utility>

#include <Id.h>
#include <Sex.h>
#include <Person.h>
#include <Phenotype.h>
#include <PhenotypeCoding.h>
#include <Configuration.h>
#include <FileReaderException.h>
#include <PersonHandler.h>
#include <PersonHandlerFactory.h>

#ifdef PROFILE
#include <boost/chrono/chrono_io.hpp>
#endif

namespace CuEira {
namespace FileIO {
class FamReaderTest;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class FamReader {
  friend FamReaderTest;
  FRIEND_TEST(FamReaderTest, StringToSex);
  FRIEND_TEST(FamReaderTest, StringToSexException);
  FRIEND_TEST(FamReaderTest, StringToPhenotypeOneTwoCoding);
  FRIEND_TEST(FamReaderTest, StringToPhenotypeZeroOneCoding);
  FRIEND_TEST(FamReaderTest, StringToPhenotypeOneTwoCodingException);
  FRIEND_TEST(FamReaderTest, StringToPhenotypeZeroOneCodingException);
public:
  explicit FamReader(const Configuration& configuration, const PersonHandlerFactory* personHandlerFactory);
  virtual ~FamReader();

  virtual PersonHandler* readPersonInformation() const;
  virtual int getNumberOfIndividualsTotal() const;

private:
  void readBasicFileInformation();
  Phenotype stringToPhenotype(std::string phenotypeString) const;
  Sex stringToSex(std::string sexString) const;

  const Configuration& configuration;
  const PersonHandlerFactory* personHandlerFactory;
  const std::string famFileStr;
  const PhenotypeCoding phenotypeCoding;
  int numberOfIndividualsTotal;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* FAMREADER_H_ */
