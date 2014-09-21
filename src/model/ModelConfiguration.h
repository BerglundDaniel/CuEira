#ifndef MODELCONFIGURATION_H_
#define MODELCONFIGURATION_H_

namespace CuEira {
namespace Model {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ModelConfiguration {
public:
  ModelConfiguration();
  virtual ~ModelConfiguration();

  ModelConfiguration(const ModelConfiguration&) = delete;
  ModelConfiguration(ModelConfiguration&&) = delete;
  ModelConfiguration& operator=(const ModelConfiguration&) = delete;
  ModelConfiguration& operator=(ModelConfiguration&&) = delete;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* MODELCONFIGURATION_H_ */
