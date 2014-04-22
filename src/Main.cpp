#include <iostream>
#include <stdexcept>

#include "Configuration.h"
#include "BimReader.h"

/**
 * This is ...
 *
 * @author Daniel Berglund daniel.k.berglund@gmail.com
 */
int main(int argc, char* argv[]) {
  CuEira::Configuration configuration = CuEira::Configuration(argc, argv);
  CuEira::FileIO::BimReader bimreader =CuEira::FileIO::BimReader(configuration);
}
