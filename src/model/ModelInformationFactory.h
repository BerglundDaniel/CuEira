#ifndef MODELINFORMATIONFACTORY_H_
#define MODELINFORMATIONFACTORY_H_

#include <ostream>

#include <ModelInformation.h>
#include <ModelState.h>
#include <SNP.h>
#include <EnvironmentFactor.h>
#include <AlleleStatistics.h>
#include <ContingencyTable.h>

namespace CuEira {
namespace Model {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ModelInformationFactory {
public:
  ModelInformationFactory();
  virtual ~ModelInformationFactory();

  virtual ModelInformation* constructModelInformation(ModelState modelState) const;

  virtual ModelInformation* constructModelInformation(ModelState modelState, const SNP& snp,
      const EnvironmentFactor& environmentFactor, const AlleleStatistics& alleleStatistics) const;

  virtual ModelInformation* constructModelInformation(ModelState modelState, const SNP& snp,
      const EnvironmentFactor& environmentFactor, const AlleleStatistics& alleleStatistics,
      const ContingencyTable& contingencyTable) const;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* MODELINFORMATIONFACTORY_H_ */
