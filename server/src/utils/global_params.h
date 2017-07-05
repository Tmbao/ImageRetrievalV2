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

struct GlobalParameters {
  bool overwrite;
  size_t nThreads;
  
  GlobalParameters() {
    overwrite = false;
    nThreads = boost::thread::hardware_concurrency();
  }
};

#endif /* global_params_h */
