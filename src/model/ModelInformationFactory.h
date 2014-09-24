#ifndef MODELINFORMATIONFACTORY_H_
#define MODELINFORMATIONFACTORY_H_

#include <ostream>

#include <ModelInformation.h>
#include <SNP.h>
#include <EnvironmentFactor.h>
#include <AlleleStatistics.h>
#include <ContingencyTable.h>
#include <ModelInformationWithTable.h>

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

  virtual ModelInformation* constructModelInformation(const SNP& snp, const EnvironmentFactor& environmentFactor,
      const AlleleStatistics& alleleStatistics) const;

  virtual ModelInformation* constructModelInformation(const SNP& snp, const EnvironmentFactor& environmentFactor,
      const AlleleStatistics& alleleStatistics, const ContingencyTable& contingencyTable) const;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* MODELINFORMATIONFACTORY_H_ */
