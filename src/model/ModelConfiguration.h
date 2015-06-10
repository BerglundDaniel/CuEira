#ifndef MODELCONFIGURATION_H_
#define MODELCONFIGURATION_H_

#include <Configuration.h>
#include <MKLWrapper.h>

namespace CuEira {
namespace Model {

using namespace CuEira::Container;

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ModelConfiguration {
public:
  virtual ~ModelConfiguration();

  virtual void setEnvironmentFactor(const HostVector& environmentData)=0;
  virtual void setSNP(const HostVector& snpData)=0;
  virtual void setInteraction(const HostVector& interactionVector)=0;

  ModelConfiguration(const ModelConfiguration&) = delete;
  ModelConfiguration(ModelConfiguration&&) = delete;
  ModelConfiguration& operator=(const ModelConfiguration&) = delete;
  ModelConfiguration& operator=(ModelConfiguration&&) = delete;

protected:
  ModelConfiguration(const Configuration& configuration);

  const Configuration& configuration;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* MODELCONFIGURATION_H_ */
