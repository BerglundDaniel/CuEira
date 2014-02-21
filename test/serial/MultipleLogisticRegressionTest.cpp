#include <cxxtest/TestSuite.h>
#include <iostream>
#include <sstream>
#include <lapackpp/gmd.h> //matrix double
#include <lapackpp/lavd.h> //vector double
#include <lapackpp/laexcp.h> //exceptions
#define private public
#include "../../src/serial/LogisticRegression.h"
#include "../../src/serial/MultipleLogisticRegression.h"

using namespace LogisticRegression;

class MultipleLogisticRegressionTestSuite: public CxxTest::TestSuite {
public:
  void testWholeFixedData(void) {
    const int numOfPredictors = 3;
    const int numOfInstances = 10;
    double *x_v, *y_v;
    LaVectorDouble* betaCoefficients = new LaVectorDouble(numOfPredictors);
    const int MAXIT = 500;

    (*betaCoefficients)(0) = 0;
    (*betaCoefficients)(1) = 0;
    (*betaCoefficients)(2) = 0;

    x_v = new double[numOfInstances * (numOfPredictors - 1)];
    y_v = new double[numOfInstances];

    //Some random generate data
    //Col1
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

    //Col2
    *(x_v + 10) = 1.4961856;
    *(x_v + 11) = -0.99393901;
    *(x_v + 12) = 1.16772209;
    *(x_v + 13) = 0.49370225;
    *(x_v + 14) = 0.76115578;
    *(x_v + 15) = -0.38176981;
    *(x_v + 16) = -0.92562295;
    *(x_v + 17) = -0.60920825;
    *(x_v + 18) = -0.62394504;
    *(x_v + 19) = 0.32976581;

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

    const LaGenMatDouble predictor = LaGenMatDouble(x_v, numOfInstances, numOfPredictors - 1);
    const LaVectorDouble binaryOutcomes = LaVectorDouble(y_v, numOfInstances);

    MultipleLogisticRegression multiplelogisticregression = MultipleLogisticRegression(predictor, binaryOutcomes,
        betaCoefficients, MAXIT);

    multiplelogisticregression.calculate();

    const LaVectorDouble& betaOut = multiplelogisticregression.getBeta();

    TS_ASSERT_DELTA(betaOut(0), 0.49203121, 1e-4);
    TS_ASSERT_DELTA(betaOut(1), 0.75177816, 1e-4);
    TS_ASSERT_DELTA(betaOut(2), -0.4946046, 1e-4);

    delete betaCoefficients;
    delete[] x_v;
    delete[] y_v;
  }

  void testExtendPredictor(void) {
    const int numOfInstances = 3;
    const int numOfPredictors = 3;
    double *x_v, *y_v;
    LaVectorDouble* betaCoefficients = new LaVectorDouble(numOfPredictors);
    const int MAXIT = 500;

    //These doesn't matter for this test
    (*betaCoefficients)(0) = 1;
    (*betaCoefficients)(1) = 2;
    (*betaCoefficients)(2) = 3;

    x_v = new double[numOfInstances * (numOfPredictors - 1)];
    y_v = new double[numOfInstances];

    //Col1
    *(x_v) = 0;
    *(x_v + 1) = 2;
    *(x_v + 2) = 3;

    //Col2
    *(x_v + 3) = 4;
    *(x_v + 4) = 5;
    *(x_v + 5) = 6;

    //These doesn't matter for this test
    *(y_v) = 1;
    *(y_v + 1) = 1;
    *(y_v + 2) = 0;

    const LaGenMatDouble predictor = LaGenMatDouble(x_v, numOfInstances, numOfPredictors - 1);
    const LaVectorDouble binaryOutcomes = LaVectorDouble(y_v, numOfInstances);
    LaVectorDouble probabilites = LaVectorDouble(numOfInstances);

    MultipleLogisticRegression multiplelogisticregression = MultipleLogisticRegression(predictor, binaryOutcomes,
        betaCoefficients, MAXIT);

    //Another column should have been added to this pointer
    const LaGenMatDouble* predictorPointer = multiplelogisticregression.predictors;

    TS_ASSERT_EQUALS((*predictorPointer)(0, 0), 1);
    TS_ASSERT_EQUALS((*predictorPointer)(1, 0), 1);
    TS_ASSERT_EQUALS((*predictorPointer)(2, 0), 1);

    TS_ASSERT_EQUALS((*predictorPointer)(0, 1), 0);
    TS_ASSERT_EQUALS((*predictorPointer)(1, 1), 2);
    TS_ASSERT_EQUALS((*predictorPointer)(2, 1), 3);

    TS_ASSERT_EQUALS((*predictorPointer)(0, 2), 4);
    TS_ASSERT_EQUALS((*predictorPointer)(1, 2), 5);
    TS_ASSERT_EQUALS((*predictorPointer)(2, 2), 6);

    delete betaCoefficients;
    delete[] x_v;
    delete[] y_v;
  }

  void testProbabilites(void) {
    const int numOfInstances = 3;
    const int numOfPredictors = 3;
    double *x_v, *y_v;
    LaVectorDouble* betaCoefficients = new LaVectorDouble(numOfPredictors);
    const int MAXIT = 500;
    double logLikelihood;
    LaVectorDouble scores = LaVectorDouble(numOfPredictors);
    LaVectorDouble probabilites = LaVectorDouble(numOfInstances);
    LaVectorDouble workVectorNx1 = LaVectorDouble(numOfInstances);

    (*betaCoefficients)(0) = 1;
    (*betaCoefficients)(1) = 2;
    (*betaCoefficients)(2) = 3;

    x_v = new double[numOfInstances * (numOfPredictors - 1)];
    y_v = new double[numOfInstances];

    //Col1
    *(x_v) = 1;
    *(x_v + 1) = 2;
    *(x_v + 2) = 0.3;

    //Col2
    *(x_v + 3) = 0.1;
    *(x_v + 4) = 0.2;
    *(x_v + 5) = 0.5;

    //These doesn't matter for this test
    *(y_v) = 1;
    *(y_v + 1) = 1;
    *(y_v + 2) = 0;

    const LaGenMatDouble predictor = LaGenMatDouble(x_v, numOfInstances, numOfPredictors - 1);
    const LaVectorDouble binaryOutcomes = LaVectorDouble(y_v, numOfInstances);

    MultipleLogisticRegression multiplelogisticregression = MultipleLogisticRegression(predictor, binaryOutcomes,
        betaCoefficients, MAXIT);

    multiplelogisticregression.calculateProbabilitiesScoreAndLogLikelihood(probabilites, scores, logLikelihood,
        betaCoefficients, workVectorNx1);

    TS_ASSERT_DELTA(probabilites(0), 0.9644288, 1e-5);
    TS_ASSERT_DELTA(probabilites(1), 0.9963157, 1e-5);
    TS_ASSERT_DELTA(probabilites(2), 0.9568927, 1e-5);

    delete betaCoefficients;
    delete[] x_v;
    delete[] y_v;
  }

  void testScore(void) {
    const int numOfInstances = 3;
    const int numOfPredictors = 3;
    double *x_v, *y_v;
    LaVectorDouble* betaCoefficients = new LaVectorDouble(numOfPredictors);
    LaVectorDouble workVectorNx1 = LaVectorDouble(numOfInstances);
    LaVectorDouble scores = LaVectorDouble(numOfPredictors);
    LaVectorDouble probabilites = LaVectorDouble(numOfInstances);
    const int MAXIT = 500;
    double logLikelihood;

    (*betaCoefficients)(0) = 1;
    (*betaCoefficients)(1) = 2;
    (*betaCoefficients)(2) = 3;

    x_v = new double[numOfInstances * (numOfPredictors - 1)];
    y_v = new double[numOfInstances];

    //Col1
    *(x_v) = 1;
    *(x_v + 1) = 2;
    *(x_v + 2) = 0.3;

    //Col2
    *(x_v + 3) = 0.1;
    *(x_v + 4) = 0.2;
    *(x_v + 5) = 0.5;

    *(y_v) = 1;
    *(y_v + 1) = 1;
    *(y_v + 2) = 0;

    const LaGenMatDouble predictor = LaGenMatDouble(x_v, numOfInstances, numOfPredictors - 1);
    const LaVectorDouble binaryOutcomes = LaVectorDouble(y_v, numOfInstances);

    MultipleLogisticRegression multiplelogisticregression = MultipleLogisticRegression(predictor, binaryOutcomes,
        betaCoefficients, MAXIT);

    multiplelogisticregression.calculateProbabilitiesScoreAndLogLikelihood(probabilites, scores, logLikelihood,
        betaCoefficients, workVectorNx1);

    TS_ASSERT_DELTA(scores(0), -0.9176373, 1e-5);
    TS_ASSERT_DELTA(scores(1), -0.2441281, 1e-5);
    TS_ASSERT_DELTA(scores(2), -0.4741524, 1e-5);

    delete betaCoefficients;
    delete[] x_v;
    delete[] y_v;
  }

  void testLogLikelihood(void) {
    const int numOfInstances = 3;
    const int numOfPredictors = 3;
    double *x_v, *y_v;
    LaVectorDouble* betaCoefficients = new LaVectorDouble(numOfPredictors);
    const int MAXIT = 500;
    double logLikelihood;
    LaVectorDouble scores = LaVectorDouble(numOfPredictors);
    LaVectorDouble workVectorNx1 = LaVectorDouble(numOfInstances);
    LaVectorDouble probabilites = LaVectorDouble(numOfInstances);

    (*betaCoefficients)(0) = 1;
    (*betaCoefficients)(1) = 2;
    (*betaCoefficients)(2) = 3;

    x_v = new double[numOfInstances * (numOfPredictors - 1)];
    y_v = new double[numOfInstances];

    //Col1
    *(x_v) = 1;
    *(x_v + 1) = 2;
    *(x_v + 2) = 0.3;

    //Col2
    *(x_v + 3) = 0.1;
    *(x_v + 4) = 0.2;
    *(x_v + 5) = 0.5;

    *(y_v) = 1;
    *(y_v + 1) = 1;
    *(y_v + 2) = 0;

    const LaGenMatDouble predictor = LaGenMatDouble(x_v, numOfInstances, numOfPredictors - 1);
    const LaVectorDouble binaryOutcomes = LaVectorDouble(y_v, numOfInstances);

    MultipleLogisticRegression multiplelogisticregression = MultipleLogisticRegression(predictor, binaryOutcomes,
        betaCoefficients, MAXIT);

    multiplelogisticregression.calculateProbabilitiesScoreAndLogLikelihood(probabilites, scores, logLikelihood,
        betaCoefficients, workVectorNx1);

    TS_ASSERT_DELTA(logLikelihood, -3.18397427, 1e-5);

    delete betaCoefficients;
    delete[] x_v;
    delete[] y_v;
  }

  void testInformationMatrix(void) {
    const int numOfInstances = 3;
    const int numOfPredictors = 3;
    double *x_v, *y_v, *p_v;
    LaVectorDouble* betaCoefficients = new LaVectorDouble(numOfPredictors);
    const int MAXIT = 500;
    double logLikelihood;
    LaVectorDouble scores = LaVectorDouble(numOfPredictors);
    LaVectorDouble workVectorNx1 = LaVectorDouble(numOfInstances);
    LaGenMatDouble informationMatrix = LaGenMatDouble(numOfPredictors, numOfPredictors);

    //These doesn't matter for this test
    (*betaCoefficients)(0) = 1;
    (*betaCoefficients)(1) = 2;
    (*betaCoefficients)(2) = 3;

    x_v = new double[numOfInstances * (numOfPredictors - 1)];
    y_v = new double[numOfInstances];
    p_v = new double[numOfInstances];

    //Col1
    *(x_v) = 1;
    *(x_v + 1) = 2;
    *(x_v + 2) = 0.3;

    //Col2
    *(x_v + 3) = 0.1;
    *(x_v + 4) = 0.2;
    *(x_v + 5) = 0.5;

    *(p_v) = 0.9;
    *(p_v + 1) = 0.3;
    *(p_v + 2) = 0.5;

    //These doesn't matter for this test
    *(y_v) = 1;
    *(y_v + 1) = 1;
    *(y_v + 2) = 0;

    const LaGenMatDouble predictor = LaGenMatDouble(x_v, numOfInstances, numOfPredictors - 1);
    const LaVectorDouble binaryOutcomes = LaVectorDouble(y_v, numOfInstances);
    LaVectorDouble probabilites = LaVectorDouble(p_v, numOfInstances);

    MultipleLogisticRegression multiplelogisticregression = MultipleLogisticRegression(predictor, binaryOutcomes,
        betaCoefficients, MAXIT);

    multiplelogisticregression.calculateInformationMatrix(informationMatrix, probabilites, workVectorNx1);

    TS_ASSERT_DELTA(informationMatrix(0, 0), 0.55, 1e-5);
    TS_ASSERT_DELTA(informationMatrix(0, 1), 0.585, 1e-5);
    TS_ASSERT_DELTA(informationMatrix(0, 2), 0.176, 1e-5);
    TS_ASSERT_DELTA(informationMatrix(1, 1), 0.9525, 1e-5);
    TS_ASSERT_DELTA(informationMatrix(1, 2), 0.1305, 1e-5);
    TS_ASSERT_DELTA(informationMatrix(2, 2), 0.0718, 1e-5);

    TS_ASSERT_EQUALS(informationMatrix(1, 0), informationMatrix(0, 1));
    TS_ASSERT_EQUALS(informationMatrix(2, 0), informationMatrix(0, 2));
    TS_ASSERT_EQUALS(informationMatrix(2, 1), informationMatrix(1, 2));

    delete betaCoefficients;
    delete[] x_v;
    delete[] y_v;
    delete[] p_v;
  }

  void testNewBeta(void) {
    const int numOfInstances = 3;
    const int numOfPredictors = 3;
    double *x_v, *y_v;
    LaVectorDouble* betaCoefficients = new LaVectorDouble(numOfPredictors);
    LaVectorDouble* betaCoefficientsOld = new LaVectorDouble(numOfPredictors);
    const int MAXIT = 500;
    double logLikelihood;
    LaVectorDouble scores = LaVectorDouble(numOfPredictors);
    LaVectorDouble workVectorNx1 = LaVectorDouble(numOfInstances);
    LaGenMatDouble informationMatrix = LaGenMatDouble(numOfPredictors, numOfPredictors);

    (*betaCoefficients)(0) = 1;
    (*betaCoefficients)(1) = 2;
    (*betaCoefficients)(2) = 3;

    (*betaCoefficientsOld)(0) = (*betaCoefficients)(0);
    (*betaCoefficientsOld)(1) = (*betaCoefficients)(1);
    (*betaCoefficientsOld)(2) = (*betaCoefficients)(2);

    scores(0) = 1;
    scores(1) = 0.5;
    scores(2) = 0.2;

    informationMatrix(0, 0) = 1;
    informationMatrix(0, 1) = 2;
    informationMatrix(0, 2) = 3;
    informationMatrix(1, 1) = 4;
    informationMatrix(1, 2) = 5;
    informationMatrix(2, 2) = 6;

    informationMatrix(1, 0) = informationMatrix(0, 1);
    informationMatrix(2, 0) = informationMatrix(0, 2);
    informationMatrix(2, 1) = informationMatrix(1, 2);

    x_v = new double[numOfInstances * (numOfPredictors - 1)];
    y_v = new double[numOfInstances];

    //These doesnt matter for this test
    //Col1
    *(x_v) = 1;
    *(x_v + 1) = 2;
    *(x_v + 2) = 0.3;

    //These doesnt matter for this test
    //Col2
    *(x_v + 3) = 0.1;
    *(x_v + 4) = 0.2;
    *(x_v + 5) = 0.5;

    //These doesnt matter for this test
    *(y_v) = 1;
    *(y_v + 1) = 1;
    *(y_v + 2) = 0;

    const LaGenMatDouble predictor = LaGenMatDouble(x_v, numOfInstances, numOfPredictors - 1);
    const LaVectorDouble binaryOutcomes = LaVectorDouble(y_v, numOfInstances);

    MultipleLogisticRegression multiplelogisticregression = MultipleLogisticRegression(predictor, binaryOutcomes,
        betaCoefficients, MAXIT);

    multiplelogisticregression.calculateNewBeta(betaCoefficients, betaCoefficientsOld, informationMatrix, scores);

    TS_ASSERT_DELTA((*betaCoefficients)(0), 0.9, 1e-5);
    TS_ASSERT_DELTA((*betaCoefficients)(1), 0.3, 1e-5);
    TS_ASSERT_DELTA((*betaCoefficients)(2), 4.5, 1e-5);

    delete betaCoefficients;
    delete betaCoefficientsOld;
    delete[] x_v;
    delete[] y_v;
  }

private:
};
