#ifndef CPUSNPVECTORFACTORY_H_
#define CPUSNPVECTORFACTORY_H_

#include <SNPVectorFactory.h>
#include <RegularHostVector.h>
#include <SNPVector.h>
#include <CpuSNPVector.h>
#include <Configuration.h>

namespace CuEira {
namespace Container {
namespace CPU {

/*
 * This class ...
 *
 *  @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuSNPVectorFactory: public SNPVectorFactory<RegularHostVector, RegularHostVector> {
public:
  explicit CpuSNPVectorFactory(const Configuration& configuration);
  virtual ~CpuSNPVectorFactory();

  virtual CpuSNPVector* constructSNPVector(SNP& snp, RegularHostVector* originalSNPData,
      const std::set<int>* snpMissingData) const;

  CpuSNPVectorFactory(const CpuSNPVectorFactory&) = delete;
  CpuSNPVectorFactory(CpuSNPVectorFactory&&) = delete;
  CpuSNPVectorFactory& operator=(const CpuSNPVectorFactory&) = delete;
  CpuSNPVectorFactory& operator=(CpuSNPVectorFactory&&) = delete;

protected:

};

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CPUSNPVECTORFACTORY_H_ */
