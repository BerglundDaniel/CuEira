#ifndef MODELINFORMATION_H_
#define MODELINFORMATION_H_

#include <string>

#include <SNP.h>
#include <EnvironmentFactor.h>
#include <AlleleStatistics.h>

namespace CuEira {
namespace Model {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ModelInformation {
  friend std::ostream& operator<<(std::ostream& os, const ModelInformation& modelInformation);
public:
  explicit ModelInformation(const SNP& snp, const EnvironmentFactor& environmentFactor,
      const AlleleStatistics& alleleStatistics);
  virtual ~ModelInformation();

  ModelInformation(const ModelInformation&) = delete;
  ModelInformation(ModelInformation&&) = delete;
  ModelInformation& operator=(const ModelInformation&) = delete;
  ModelInformation& operator=(ModelInformation&&) = delete;

protected:
  explicit ModelInformation(); //For the mock
  virtual void toOstream(std::ostream& os) const;

  const SNP* snp;
  const EnvironmentFactor* environmentFactor;
  const AlleleStatistics* alleleStatistics;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* MODELINFORMATION_H_ */
