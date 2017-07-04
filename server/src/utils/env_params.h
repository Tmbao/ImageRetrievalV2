//
//  env_params.h
//  server
//
//  Created by Bao Truong on 7/4/17.
//
//

#ifndef env_params_h
#define env_params_h

#include <boost/thread.hpp>

namespace env_params {
  
  bool overwrite() {
    return false;
  }
  
  size_t no_threads() {
    return boost::thread::hardware_concurrency();
  }
};

#endif /* env_params_h */
