#ifndef MODELHANDLER_H_
#define MODELHANDLER_H_

#include <DataHandler.h>
#include <Statistics.h>

namespace CuEira {
namespace Model {

/**
 * This is
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ModelHandler {
public:
  ModelHandler(Container::DataHandler& dataHandler);
  virtual ~ModelHandler();

  bool next();
  void calculateModel();
  Statistics getStatistics() const;

protected:
  Container::DataHandler& dataHandler;
};

} /* namespace Model */
} /* namespace CuEira */

#endif /* MODELHANDLER_H_ */
