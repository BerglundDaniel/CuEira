#ifndef SNPVECTORFACTORY_H_
#define SNPVECTORFACTORY_H_

#include <vector>
#include <set>
#include <type_traits>
#include <typeinfo>

#include <SNPVector.h>
#include <SNP.h>
#include <Configuration.h>
#include <GeneticModel.h>
#include <HostVector.h>

#ifndef CPU
#include <DeviceVector.h>
#include <PinnedHostVector.h>
#endif

namespace CuEira {
namespace Container {

/**
 * This class....
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
template<typename Vector>
class SNPVectorFactory {
public:
  virtual ~SNPVectorFactory();

  template<typename U = Vector,
  typename std::enable_if<!(typeid(U)==typeid(DeviceVector))>::type* = nullptr>
  SNPVector<U>* constructSNPVector(SNP& snp, U* originalSNPData, const std::set<int>* snpMissingData) const;
  //{
  //}

  template<typename U = Vector,
  typename std::enable_if<(typeid(U)==typeid(DeviceVector))>::type* = nullptr>
  SNPVector<U>* constructSNPVector(SNP& snp, PinnedHostVector* originalSNPData, const std::set<int>* snpMissingData) const;
  //{
  //}

  SNPVectorFactory(const SNPVectorFactory&) = delete;
  SNPVectorFactory(SNPVectorFactory&&) = delete;
  SNPVectorFactory& operator=(const SNPVectorFactory&) = delete;
  SNPVectorFactory& operator=(SNPVectorFactory&&) = delete;

protected:
  explicit SNPVectorFactory(const Configuration& configuration);

  void updateSize(Vector* originalSNPData, const std::set<int>* snpMissingData) const;

  const Configuration& configuration;
  const GeneticModel geneticModel;
};

} /* namespace Container */
} /* namespace CuEira */

#endif /* SNPVECTORFACTORY_H_ */
