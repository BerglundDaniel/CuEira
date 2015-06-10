#ifndef KERNELWRAPPER_H_
#define KERNELWRAPPER_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <sstream>
#include <string>

#include <CudaAdapter.cu>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <CudaException.h>
#include <Stream.h>

namespace CuEira {
namespace CUDA {
namespace Kernel {

using namespace CuEira::Container;

/**
 * This is a class that wraps kernels
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

const static int numberOfThreadsPerBlock = 256; //FIXME

/**
 * Performs the logistic transform, exp(x)/(1+exp(x)), on each element in logitVector and stores the results in probabilities
 */
void logisticTransform(const Stream& stream, const DeviceVector& logitVector, DeviceVector& probabilites);

/**
 * Divides each element in numeratorVector with its corresponding element in denomitorVector
 */
void elementWiseDivision(const Stream& stream, const DeviceVector& numeratorVector, const DeviceVector& denomitorVector,
    DeviceVector& result);

/**
 * Adds each element result(i)=vector1(i) + vector2(i)
 */
void elementWiseAddition(const Stream& stream, const DeviceVector& vector1, const DeviceVector& vector2,
    DeviceVector& result);

/**
 * Multiplies each element result(i)=vector1(i) * vector2(i)
 */
void elementWiseMultiplication(const Stream& stream, const DeviceVector& vector1, const DeviceVector& vector2,
    DeviceVector& result);

/**
 * Calculates all the parts of a loglikelihood. The sum of the elements in result is the loglikelihood
 */
void logLikelihoodParts(const Stream& stream, const DeviceVector& outcomesVector, const DeviceVector& probabilites,
    DeviceVector& result);

/**
 * Calculates the absolute difference for each element
 */
void elementWiseAbsoluteDifference(const Stream& stream, const DeviceVector& vector1, const DeviceVector& vector2,
    DeviceVector& result);

/**
 * Calculates x*(1-x) for each element
 */
void probabilitesMultiplyProbabilites(const Stream& stream, const DeviceVector& probabilitesDevice,
    DeviceVector& result);

/**
 * Calculates vector1-vector2
 */
void elementWiseDifference(const Stream& stream, const DeviceVector& vector1, const DeviceVector& vector2,
    DeviceVector& result);

//to=from[indexes[i]]
void vectorCopyIndexes(const Stream& stream, const DeviceVector& indexes, const DeviceVector& from, DeviceVector& to);

//c-vector[i]
void constSubtractVector(const Stream& stream, const int c, DeviceVector& vector);

//to[i] = snpToRisk[from[i]]
void applyGeneticModel(const Stream& stream, const int snpToRisk[3], const DeviceVector& from, DeviceVector& to);

//interaction=vector1*vector2 vector1=vector2=0 if interaction!=0
void applyAdditiveModel(const Stream& stream, DeviceVector& vector1, DeviceVector& vector2, DeviceVector& interaction);

//numberOfAllelesPerGenotype=snpData(i)+3*phenotypeData(i)
Container::DeviceMatrix* calculateNumberOfAllelesPerGenotype(const Stream& stream,
    const Container::DeviceVector& snpData, const Container::DeviceVector& phenotypeData);

} /* namespace Kernel */
} /* namespace CUDA */
} /* namespace CuEira */

#endif /* KERNELWRAPPER_H_ */
