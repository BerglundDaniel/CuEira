#ifndef GPUWORKERTHREAD_H_
#define GPUWORKERTHREAD_H_

#include <thread>

#include <Configuration.h>
#include <Device.h>
#include <DataHandlerFactory.h>
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
#include <ResultWriter.h>

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
    const DataHandlerFactory* dataHandlerFactory, FileIO::ResultWriter* resultWriter);

} /* namespace CUDA */
} /* namespace CuEira */

#endif /* GPUWORKERTHREAD_H_ */