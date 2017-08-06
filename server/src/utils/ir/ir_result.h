//
//  ir_result.h
//  server
//
//  Created by Bao Truong on 7/3/17.
//
//

#ifndef ir_result_h
#define ir_result_h

#include <json.hpp>
#include <string>


namespace ir {

class IrResult {
 private:
  std::string name_;
  float score_;

 public:
  IrResult(std::string name, float score):
    name_(name), score_(score) {}

  inline std::string name() const {
    return name_;
  }

  inline float score() const {
    return score_;
  }
};

inline void to_json(nlohmann::json &j, const IrResult &result) {
  j = nlohmann::json{{"name", result.name()}, {"score", result.score()}};
}

}

#endif /* ir_result_h */
