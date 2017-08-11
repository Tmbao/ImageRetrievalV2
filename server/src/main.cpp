#include <json.hpp>
#include <silicon/api.hh>
#include <silicon/backends/mhd.hh>

#include "utils/ir/ir_instance.h"
#include "server_configurations.h"

using namespace sl;
using namespace s;

#ifndef IOD_SYMBOL_processImage
#define IOD_SYMBOL_processImage
    iod_define_symbol(processImage)
#endif

auto ir_api = http_api(
  POST / _processImage * post_parameters(_path = std::string()) = [] (auto param) {
    cv::Mat image = cv::imread(param.path);
    std::vector<ir::IrResult> result = ir::IrInstance::retrieve(image);
    
    nlohmann::json j(result);
    return j.dump();
  }
);

int main() {
  ir::IrInstance::createInstanceIfNecessary(
    GlobalParams(
      false,
      configs::BATCH_SIZE,
      configs::N_THREADS),
    ir::QuantizationParams(
      8,
      3,
      800,
      6250,
      configs::CODEBOOK_FILE,
      configs::CODEBOOK_NAME,
      configs::INDEX_FILE
    ),
    ir::DatabaseParams(
      configs::N_WORDS,
      configs::IMAGE_FOLDER,
      configs::CACHE_FOLDER
    ));
  mhd_json_serve(ir_api, 1235);
}