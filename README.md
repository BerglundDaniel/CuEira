CuEira
======

Gene-Enviroment interaction analysis in CUDA

Uses Boost, MKL, CUDA and CUBLAS
CPU version uses Lapackpp, this will likely be changed later. CPU version is broken at the moment.
GPU needs MKL and Intel compiler newer than 13.1

Make sure you have atleast cmake 2.8 and cuda 5.5

You might have to set some paths by using the CMakes -D option or EXPORT BOOST_ROOT=path_to_boost
