//
//  quantization_params.h
//  server
//
//  Created by Bao Truong on 7/3/17.
//
//

#ifndef quantization_params_h
#define quantization_params_h

#include <string>


namespace ir {

/**
 * A parameter type used for quantization step.
 */
struct QuantizationParams {
  int nTrees;
  size_t knn;
  int nChecks;
  float deltaSqr;
  std::string codebookFile;
  std::string codebookName;
  std::string indexFile;

  QuantizationParams() {
    nTrees = 8;
    knn = 3;
    nChecks = 800;
    deltaSqr = 6250;
    //TODO: Update these
    codebookFile = "";
    codebookName = "";
    indexFile = "";
  }
};
  
}

#endif /* quantization_params_h */
