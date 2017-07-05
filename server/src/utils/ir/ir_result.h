//
//  ir_result.h
//  server
//
//  Created by Bao Truong on 7/3/17.
//
//

#ifndef ir_result_h
#define ir_result_h

#include <string>


namespace ir {

class IrResult {
 private:
  std::string name_;
  double score_;

 public:
  IrResult(std::string name, double score):
    name_(name), score_(score) {}

  std::string name() {
    return name_;
  }

  double score() {
    return score_;
  }
};

}

#endif /* ir_result_h */
