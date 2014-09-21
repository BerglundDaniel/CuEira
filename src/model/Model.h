#ifndef MODEL_H_
#define MODEL_H_

#include <ModelResult.h>

namespace CuEira {
namespace Model {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class Model {
public:
  Model();
  virtual ~Model();

  virtual ModelResult* calculate()=0;

  Model(const Model&) = delete;
  Model(Model&&) = delete;
  Model& operator=(const Model&) = delete;
  Model& operator=(Model&&) = delete;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* MODEL_H_ */
