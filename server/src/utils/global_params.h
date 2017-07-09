//
//  global_params.h
//  server
//
//  Created by Bao Truong on 7/4/17.
//
//

#ifndef global_params_h
#define global_params_h

#include <boost/thread.hpp>

struct GlobalParams {
  bool overwrite;
  size_t nThreads;

  GlobalParams(
    bool overwrite = false,
    size_t nThreads = boost::thread::hardware_concurrency()):
    overwrite(overwrite), nThreads(nThreads) {}
};

#endif /* global_params_h */
