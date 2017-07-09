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
  size_t nTrees;
  size_t knn;
  size_t nChecks;
  float deltaSqr;
  std::string codebookFile;
  std::string codebookName;
  std::string indexFile;

  QuantizationParams(
    size_t nTrees = 8,
    size_t knn = 3,
    size_t nChecks = 800,
    float deltaSqr = 6250,
    std::string codebookFile = "",
    std::string codebookName = "",
    std::string indexFile = ""):
    nTrees(nTrees),
    knn(knn),
    nChecks(nChecks),
    deltaSqr(deltaSqr),
    codebookFile(codebookFile),
    codebookName(codebookName),
    indexFile(indexFile) {}
};

}

#endif /* quantization_params_h */
