#ifndef CPUWORKERTHREAD_H_
#define CPUWORKERTHREAD_H_

namespace CuEira {
namespace CPU {

/**
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */

/**
 * Thread to do work using CPU
 */
void CPUWorkerThread(const Configuration* configuration, const DataHandlerFactory* dataHandlerFactory,
    FileIO::ResultWriter* resultWriter);

} /* namespace CPU */
} /* namespace CuEira */

#endif /* CPUWORKERTHREAD_H_ */
