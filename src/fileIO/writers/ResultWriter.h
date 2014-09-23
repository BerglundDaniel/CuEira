#ifndef RESULTWRITER_H_
#define RESULTWRITER_H_

#include <fstream>
#include <thread>
#include <mutex>
#include <sstream>

#include <Configuration.h>
#include <FileReaderException.h>
#include <ModelInformation.h>
#include <CombinedResults.h>

#ifdef PROFILE
#include <boost/chrono/chrono_io.hpp>
#endif

namespace CuEira {
namespace FileIO {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class ResultWriter {
public:
  ResultWriter(const Configuration& configuration);
  virtual ~ResultWriter();

  void writeFullResult(const Model::ModelInformation* modelInformation, const Model::CombinedResults* combinedResults);
  void writePartialResult(const Model::ModelInformation* modelInformation);

  ResultWriter(const ResultWriter&) = delete;
  ResultWriter(ResultWriter&&) = delete;
  ResultWriter& operator=(const ResultWriter&) = delete;
  ResultWriter& operator=(ResultWriter&&) = delete;

private:
  void printHeader();
  void openFile();
  void closeFile();

  const Configuration& configuration;
  std::ofstream outputStream;
  std::string outputFileName;
  std::mutex fileLock;
#ifdef PROFILE
  boost::chrono::duration<double> timeWaitTotalLock;
#endif
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* RESULTWRITER_H_ */
