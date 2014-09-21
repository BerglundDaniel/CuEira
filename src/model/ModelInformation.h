#ifndef MODELINFORMATION_H_
#define MODELINFORMATION_H_

#include <string>

#include <ModelState.h>

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
  explicit ModelInformation(ModelState modelState, std::string information);
  virtual ~ModelInformation();

  virtual ModelState getModelState() const;

  ModelInformation(const ModelInformation&) = delete;
  ModelInformation(ModelInformation&&) = delete;
  ModelInformation& operator=(const ModelInformation&) = delete;
  ModelInformation& operator=(ModelInformation&&) = delete;

protected:
  virtual void toOstream(std::ostream& os) const;

private:
  ModelState modelState;
  std::string information;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* MODELINFORMATION_H_ */
