#include <cxxtest/TestSuite.h>
#include <iostream>
#include <sstream>
#include <math.h>
#include <lapackpp/gmd.h> //matrix double
#include <lapackpp/lavd.h> //vector double
#include <lapackpp/laexcp.h> //exceptions
#define private public
#include "../../src/serial/LogisticRegression.h"
#include "../../src/serial/SimpleLogisticRegression.h"

using namespace LogisticRegression;

class SimpleLogisticRegressionTestSuite: public CxxTest::TestSuite {
public:
  void testWholeFixedData(void) {
    const int numOfInstances = 10;
    double *x_v, *y_v;
    LaVectorDouble* betaCoefficients = new LaVectorDouble(2);
    const int MAXIT = 500;

    (*betaCoefficients)(0) = 0;
    (*betaCoefficients)(1) = 0;

    x_v = new double[numOfInstances];
    y_v = new double[numOfInstances];

    //Some random generate data
    *(x_v) = 1.33582291;
    *(x_v + 1) = -0.21913482;
    *(x_v + 2) = 0.29749252;
    *(x_v + 3) = 0.49347861;
    *(x_v + 4) = -0.57089565;
    *(x_v + 5) = -1.03339458;
    *(x_v + 6) = 0.11693107;
    *(x_v + 7) = -0.38543587;
    *(x_v + 8) = 0.25468775;
    *(x_v + 9) = -0.69603999;

    *(y_v) = 1;
    *(y_v + 1) = 1;
    *(y_v + 2) = 0;
    *(y_v + 3) = 0;
    *(y_v + 4) = 1;
    *(y_v + 5) = 0;
    *(y_v + 6) = 1;
    *(y_v + 7) = 0;
    *(y_v + 8) = 1;
    *(y_v + 9) = 1;

    const LaVectorDouble predictor = LaVectorDouble(x_v, numOfInstances);
    const LaVectorDouble binaryOutcomes = LaVectorDouble(y_v, numOfInstances);

    SimpleLogisticRegression simplelogisticregression = SimpleLogisticRegression(predictor, binaryOutcomes,
        betaCoefficients, MAXIT);

    simplelogisticregression.calculate();

    const LaVectorDouble& betaOut = simplelogisticregression.getBeta();

    TS_ASSERT_DELTA(betaOut(0), 0.4349, 1e-4);
    TS_ASSERT_DELTA(betaOut(1), 0.4720, 1e-4);

    delete betaCoefficients;
    delete[] x_v;
    delete[] y_v;
  }

  void testProbabilites(void) {
    const int numOfInstances = 3;
    double *x_v, *y_v;
    LaVectorDouble* betaCoefficients = new LaVectorDouble(2);
    const int MAXIT = 500;

    (*betaCoefficients)(0) = 1;
    (*betaCoefficients)(1) = 2;

    x_v = new double[numOfInstances];
    y_v = new double[numOfInstances];

    *(x_v) = 1;
    *(x_v + 1) = 2;
    *(x_v + 2) = 0.3;

    //These doesn't matter for this test
    *(y_v) = 1;
    *(y_v + 1) = 1;
    *(y_v + 2) = 0;

    const LaVectorDouble predictor = LaVectorDouble(x_v, numOfInstances);
    const LaVectorDouble binaryOutcomes = LaVectorDouble(y_v, numOfInstances);
    LaVectorDouble probabilites = LaVectorDouble(numOfInstances);

    SimpleLogisticRegression simplelogisticregression = SimpleLogisticRegression(predictor, binaryOutcomes,
        betaCoefficients, MAXIT);

    simplelogisticregression.calculateProbabilities(probabilites, betaCoefficients);

    TS_ASSERT_DELTA(probabilites(0), 0.952574, 1e-5);
    TS_ASSERT_DELTA(probabilites(1), 0.993307, 1e-5);
    TS_ASSERT_DELTA(probabilites(2), 0.832018, 1e-5);

    delete betaCoefficients;
    delete[] x_v;
    delete[] y_v;
  }

  void testScoreAndLogLikelihood(void) {
    const int numOfInstances = 3;
    double *x_v, *y_v, *prob_v;
    LaVectorDouble* betaCoefficients = new LaVectorDouble(2);
    const int MAXIT = 500;
    LaVectorDouble scores = LaVectorDouble(betaCoefficients->size());
    double logLikelihood = 0;

    (*betaCoefficients)(0) = 1;
    (*betaCoefficients)(1) = 2;

    x_v = new double[numOfInstances];
    y_v = new double[numOfInstances];
    prob_v = new double[numOfInstances];

    *(x_v) = 1;
    *(x_v + 1) = 2;
    *(x_v + 2) = 0.3;

    *(y_v) = 1;
    *(y_v + 1) = 0;
    *(y_v + 2) = 1;

    *(prob_v) = 0.9;
    *(prob_v + 1) = 0.3;
    *(prob_v + 2) = 0.5;

    const LaVectorDouble predictor = LaVectorDouble(x_v, numOfInstances);
    const LaVectorDouble binaryOutcomes = LaVectorDouble(y_v, numOfInstances);
    const LaVectorDouble probabilites = LaVectorDouble(prob_v, numOfInstances);

    SimpleLogisticRegression simplelogisticregression = SimpleLogisticRegression(predictor, binaryOutcomes,
        betaCoefficients, MAXIT);

    simplelogisticregression.calculateScoreAndLogLikelihood(scores, logLikelihood, probabilites);

    TS_ASSERT_DELTA(scores(0), 0.2999999, 1e-5);
    TS_ASSERT_DELTA(scores(1), -0.34999999, 1e-5);

    TS_ASSERT_DELTA(logLikelihood, -1.155182, 1e-5);

    delete betaCoefficients;
    delete[] x_v;
    delete[] y_v;
    delete[] prob_v;
  }

  void testInformationMatrix(void) {
    const int numOfInstances = 3;
    double *x_v, *y_v, *prob_v;
    LaVectorDouble* betaCoefficients = new LaVectorDouble(2);
    LaGenMatDouble informationMatrix = LaGenMatDouble(2, 2);
    const int MAXIT = 500;
    double logLikelihood = 0;

    //Doesn't matter for this test
    (*betaCoefficients)(0) = 1;
    (*betaCoefficients)(1) = 2;

    x_v = new double[numOfInstances];
    y_v = new double[numOfInstances];
    prob_v = new double[numOfInstances];

    *(x_v) = 1;
    *(x_v + 1) = 2;
    *(x_v + 2) = 0.3;

    //Doesn't matter for this test
    *(y_v) = 1;
    *(y_v + 1) = 0;
    *(y_v + 2) = 1;

    *(prob_v) = 0.9;
    *(prob_v + 1) = 0.3;
    *(prob_v + 2) = 0.5;

    const LaVectorDouble predictor = LaVectorDouble(x_v, numOfInstances);
    const LaVectorDouble binaryOutcomes = LaVectorDouble(y_v, numOfInstances);
    const LaVectorDouble probabilites = LaVectorDouble(prob_v, numOfInstances);

    SimpleLogisticRegression simplelogisticregression = SimpleLogisticRegression(predictor, binaryOutcomes,
        betaCoefficients, MAXIT);

    simplelogisticregression.calculateInformationMatrix(informationMatrix, probabilites);

    TS_ASSERT_DELTA(informationMatrix(0, 0), 0.55, 1e-5);
    TS_ASSERT_DELTA(informationMatrix(0, 1), 0.585, 1e-5);
    TS_ASSERT_DELTA(informationMatrix(1, 1), 0.9525, 1e-5);
    TS_ASSERT_EQUALS(informationMatrix(0, 1), informationMatrix(1, 0));

    delete betaCoefficients;
    delete[] x_v;
    delete[] y_v;
    delete[] prob_v;
  }

  void testNewBeta(void) {
    const int numOfInstances = 3;
    double *x_v, *y_v, *prob_v;
    LaVectorDouble* betaCoefficients = new LaVectorDouble(2);
    LaVectorDouble* betaCoefficientsOld = new LaVectorDouble(2);
    LaGenMatDouble informationMatrix = LaGenMatDouble(2, 2);
    LaVectorDouble scores = LaVectorDouble(2);
    const int MAXIT = 500;
    double logLikelihood = 0;

    (*betaCoefficients)(0) = 1;
    (*betaCoefficients)(1) = 2;

    (*betaCoefficientsOld)(0) = (*betaCoefficients)(0);
    (*betaCoefficientsOld)(1) = (*betaCoefficients)(1);

    x_v = new double[numOfInstances];
    y_v = new double[numOfInstances];
    prob_v = new double[numOfInstances];

    *(x_v) = 1;
    *(x_v + 1) = 2;
    *(x_v + 2) = 0.3;

    *(y_v) = 1;
    *(y_v + 1) = 0;
    *(y_v + 2) = 1;

    scores(0) = 1;
    scores(1) = 0.5;

    informationMatrix(0, 0) = 0.3;
    informationMatrix(0, 1) = 0.5;
    informationMatrix(1, 1) = 1;
    informationMatrix(1, 0) = informationMatrix(0, 1);

    const LaVectorDouble predictor = LaVectorDouble(x_v, numOfInstances);
    const LaVectorDouble binaryOutcomes = LaVectorDouble(y_v, numOfInstances);

    SimpleLogisticRegression simplelogisticregression = SimpleLogisticRegression(predictor, binaryOutcomes,
        betaCoefficients, MAXIT);

    simplelogisticregression.calculateNewBeta(betaCoefficients, betaCoefficientsOld, informationMatrix, scores);

    TS_ASSERT_DELTA((*betaCoefficients)(0), 16.0, 1e-6);
    TS_ASSERT_DELTA((*betaCoefficients)(1), -5.0, 1e-6);

    delete betaCoefficients;
    delete betaCoefficientsOld;
    delete[] x_v;
    delete[] y_v;
    delete[] prob_v;
  }

private:
};
