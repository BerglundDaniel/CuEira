#ifndef FAMREADER_H_
#define FAMREADER_H_

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>
#include <boost/algorithm/string.hpp>

#include <Id.h>
#include <Sex.h>
#include <Person.h>
#include <PersonHandler.h>
#include <Phenotype.h>
#include <PhenotypeCoding.h>
#include <Configuration.h>
#include <FileReaderException.h>

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
  explicit FamReader(const Configuration& configuration);
  virtual ~FamReader();

  PersonHandler* readPersonInformation() const;

private:
  Phenotype stringToPhenotype(std::string phenotypeString) const;
  Sex stringToSex(std::string sexString) const;

  const Configuration& configuration;
  const std::string famFileStr;
  const PhenotypeCoding phenotypeCoding;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* FAMREADER_H_ */
