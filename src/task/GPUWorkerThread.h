#ifndef GPUWORKERTHREAD_H_
#define GPUWORKERTHREAD_H_

#include <thread>

#include <Configuration.h>
#include <Device.h>
#include <DataHandlerFactory.h>
#include <BedReader.h>
#include <DeviceVector.h>
#include <DeviceMatrix.h>
#include <HostVector.h>
#include <HostMatrix.h>
#include <DataHandler.h>
#include <Stream.h>
#include <LogisticRegressionConfiguration.h>
#include <LogisticRegression.h>
#include <GpuModelHandler.h>
#include <SNP.h>
#include <DataHandlerState.h>
#include <HostToDevice.h>
#include <DeviceToHost.h>
#include <Statistics.h>
#include <StreamFactory.h>
#include <KernelWrapperFactory.h>
#include <KernelWrapper.h>

namespace CuEira {
namespace CUDA {

/**
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

/**
 * Thread to do work on GPU
 */
void GPUWorkerThread(const Configuration* configuration, const Device* device,
    const DataHandlerFactory* dataHandlerFactory, const FileIO::BedReader* bedReader);

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* GPUWORKERTHREAD_H_ */
