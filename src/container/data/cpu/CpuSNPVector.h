#ifndef CPUSNPVECTOR_H_
#define CPUSNPVECTOR_H_

#include <set>

#include <SNPVector.h>
#include <RegularHostVector.h>
#include <SNP.h>
#include <GeneticModel.h>
#include <Recode.h>

namespace CuEira {
namespace Container {
namespace CPU {

using namespace CuEira::CPU;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class CpuSNPVector: public SNPVector<RegularHostVector> {
public:
  explicit CpuSNPVector(SNP& snp, GeneticModel geneticModel, const RegularHostVector* snpOrgExMissing,
      const std::set<int>* snpMissingData);
  virtual ~CpuSNPVector();

protected:
  virtual void doRecode(int snpToRisk[3]);
};

} /* namespace CPU */
} /* namespace Container */
} /* namespace CuEira */

#endif /* CPUSNPVECTOR_H_ */
