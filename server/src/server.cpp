// This file generated by ngrestcg
// For more information, please visit: https://github.com/loentar/ngrest

#include "server.h"

#include <json.hpp>
#include <opencv2/opencv.hpp>

#include "utils/taskmgr/task_manager.h"
#include "server_configurations.h"


server::server() {
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
}

std::string server::processImage(const std::string &path) {
  cv::Mat image = cv::imread(path);
  std::vector<ir::IrResult> result = ir::IrInstance::retrieve(image);
  
  nlohmann::json j(result);
  return j.dump();
}

std::string server::enqueueImage(const std::string& path) {
  cv::Mat img = cv::imread(path);
  if (taskmgr::TaskManager::addTask(path, img)) {
    return path;
  } else {
    return "";
  }
}

std::string server::fetchResult(const std::string& id) {
  std::vector<ir::IrResult> result;
  taskmgr::TaskStatus status = taskmgr::TaskManager::fetchResult(id, result);

  if (status == taskmgr::READY) {
    nlohmann::json j(result);
    return j.dump();
  } else {
    return "";
  }
}

std::string server::flush() {
  taskmgr::TaskManager::execute();
  return "";
}
