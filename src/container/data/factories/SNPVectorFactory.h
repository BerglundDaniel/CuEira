#ifndef SNPVECTORFACTORY_H_
#define SNPVECTORFACTORY_H_

#include <vector>
#include <set>

#include <SNPVector.h>
#include <SNP.h>
#include <Configuration.h>
#include <GeneticModel.h>
#include <HostVector.h>

namespace CuEira {
namespace Container {

/**
 * This class constructs SNPVectors and makes checks on the data to see if the SNP should be included or not.
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class SNPVectorFactory {
public:
  SNPVectorFactory(const Configuration& configuration);
  virtual ~SNPVectorFactory();

  virtual SNPVector<>* constructSNPVector(SNP& snp, const HostVector* originalSNPData, const std::set<int>* snpMissingData) const=0;

  SNPVectorFactory(const SNPVectorFactory&) = delete;
  SNPVectorFactory(SNPVectorFactory&&) = delete;
  SNPVectorFactory& operator=(const SNPVectorFactory&) = delete;
  SNPVectorFactory& operator=(SNPVectorFactory&&) = delete;

private:
  const Configuration& configuration;
  const GeneticModel geneticModel;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* SNPVECTORFACTORY_H_ */
