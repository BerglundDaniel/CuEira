#ifndef RESULTWRITER_H_
#define RESULTWRITER_H_

#include <fstream>
#include <thread>
#include <mutex>
#include <sstream>

#include <Configuration.h>
#include <Statistics.h>
#include <SNP.h>
#include <EnvironmentFactor.h>
#include <FileReaderException.h>
#include <SNPVector.h>

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

  void writeFullResult(const SNP& snp, const EnvironmentFactor& environmentFactor, const Statistics& statistics, const Container::SNPVector& snpVector);
  void writePartialResult(const SNP& snp, const EnvironmentFactor& environmentFactor);

private:
  void printHeader();
  void openFile();
  void closeFile();

  const Configuration& configuration;
  std::ofstream outputStream;
  std::string outputFileName;
  std::mutex fileLock;
};

} /* namespace FileIO */
} /* namespace CuEira */

#endif /* RESULTWRITER_H_ */
