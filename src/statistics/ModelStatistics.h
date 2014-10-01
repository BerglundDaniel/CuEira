#ifndef MODELSTATISTICS_H_
#define MODELSTATISTICS_H_

namespace CuEira {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ModelStatistics {
  friend std::ostream& operator<<(std::ostream& os, const ModelStatistics& modelStatistics);
public:
  ModelStatistics();
  virtual ~ModelStatistics();

  ModelStatistics(const ModelStatistics&) = delete;
  ModelStatistics(ModelStatistics&&) = delete;
  ModelStatistics& operator=(const ModelStatistics&) = delete;
  ModelStatistics& operator=(ModelStatistics&&) = delete;

protected:
  virtual void toOstream(std::ostream& os) const=0;
};

} /* namespace CuEira */

#endif /* MODELSTATISTICS_H_ */
