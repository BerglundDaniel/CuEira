#ifndef CONSTRUCTORHELPERS_H_
#define CONSTRUCTORHELPERS_H_

#include <sstream>
#include <string>
#include <stdlib.h>
#include <time.h>

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

#ifdef CPU
#include <lapackpp/gmd.h>
#include <LapackppHostMatrix.h>
#else
#include <PinnedHostMatrix.h>
#endif

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

  Person* constructPersonInclude(int number);
  Person* constructPersonNotInclude(int number);
  Person* constructPersonInclude(int number, Phenotype phenotype);

  Container::EnvironmentVectorMock* constructEnvironmentVectorMock();
  Container::SNPVectorMock* constructSNPVectorMock();
  EnvironmentFactorHandlerMock* constructEnvironmentFactorHandlerMock();

private:

};

} /* namespace CuEira_Test */
} /* namespace CuEira */

#endif /* CONSTRUCTORHELPERS_H_ */
