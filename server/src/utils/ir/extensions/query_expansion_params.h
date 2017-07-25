//
//  query_expansion_params.h
//  server
//
//  Created by Bao Truong on 7/24/17.
//
//

#ifndef query_expansion_params_h
#define query_expansion_params_h

namespace ir {

typedef int size_t;

/**
 * A parameter type used for average query expansion.
 */
struct QueryExpansionParams {
  size_t nChosen;
  size_t evalPeriod;

  QueryExpansionParams(size_t nChosen = 10, size_t evalPeriod = 8):
    nChosen(nChosen),
    evalPeriod(evalPeriod) {}
};

}

#endif /* query_expansion_params_h */
