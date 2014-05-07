#ifndef FAMREADER_H_
#define FAMREADER_H_

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>


#include <Id.h>
#include <Sex.h>
#include <Person.h>
#include <PersonHandler.h>
#include <Phenotype.h>
#include <PhenotypeCoding.h>
#include <Configuration.h>
#include <FileReaderException.h>

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
  explicit FamReader(const Configuration& configuration, PersonHandler& personHandler);
  virtual ~FamReader();

  const PersonHandler& getPersonHandler() const;

private:
  Phenotype stringToPhenotype(std::string phenotypeString) const;
  Sex stringToSex(std::string sexString) const;

  const Configuration& configuration;
  PersonHandler& personHandler;
  const std::string famFileStr;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* FAMREADER_H_ */
