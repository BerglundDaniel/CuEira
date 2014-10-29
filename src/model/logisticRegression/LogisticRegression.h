#ifndef LOGISTICREGRESSION_H_
#define LOGISTICREGRESSION_H_

#include <Model.h>
#include <LogisticRegressionResult.h>
#include <LogisticRegressionConfiguration.h>
#include <MKLWrapper.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <RegularHostVector.h>
#include <RegularHostMatrix.h>

namespace CuEira {
namespace Model {
namespace LogisticRegression {

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
class LogisticRegression: public Model {
public:
  virtual ~LogisticRegression();

  virtual LogisticRegressionResult* calculate()=0;

  LogisticRegression(const LogisticRegression&) = delete;
  LogisticRegression(LogisticRegression&&) = delete;
  LogisticRegression& operator=(const LogisticRegression&) = delete;
  LogisticRegression& operator=(LogisticRegression&&) = delete;

protected:
  LogisticRegression(LogisticRegressionConfiguration* logisticRegressionConfiguration);
  LogisticRegression(); //For the mock

  void invertInformationMatrix(HostMatrix& informationMatrixHost, HostMatrix& inverseInformationMatrixHost,
      HostMatrix& uSVD, HostVector& sigma, HostMatrix& vtSVD, HostMatrix& workMatrixMxMHost);
  void calculateNewBeta(HostMatrix& inverseInformationMatrixHost, HostVector& scoresHost,
      HostVector& betaCoefficentsHost);
  void calculateDifference(const HostVector& betaCoefficentsHost, HostVector& betaCoefficentsOldHost,
      PRECISION& diffSumHost);

  LogisticRegressionConfiguration* logisticRegressionConfiguration;

  const int numberOfRows;
  const int numberOfPredictors;
  const int maxIterations;
  const double convergenceThreshold;
  PRECISION logLikelihood;

  Container::RegularHostVector* betaCoefficentsOldHost;
  Container::RegularHostVector* sigma;
  Container::RegularHostMatrix* uSVD;
  Container::RegularHostMatrix* vtSVD;
  Container::RegularHostMatrix* workMatrixMxMHost;
  Container::HostVector* scoresHost; //Config class inherited from LR config owns it
};

} /* namespace LogisticRegression */
} /* namespace Model */
} /* namespace CuEira */

#endif /* LOGISTICREGRESSION_H_ */
