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
  float score_;

 public:
  IrResult(std::string name, float score):
    name_(name), score_(score) {}

  std::string name() const {
    return name_;
  }

  float score() const {
    return score_;
  }
};

}

#endif /* ir_result_h */
