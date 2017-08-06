#include <opencv2/opencv.hpp>
#include "utils/ir/ir_instance.h"

int main() {
  GlobalParams globalParams;
  ir::QuantizationParams quantParams(
    8,
    3,
    800,
    6250,
    "<file_codebook>",
    "clusters", // Codebook name
    "<file_index>");
  ir::DatabaseParams dbParams(
    1000000, // Number of words
    "<image_folder>",
    "<temp_folder>"); // Cache folder (for saving tfidf, descriptors...)
  
  ir::IrInstance::createInstanceIfNecessary(globalParams, quantParams, dbParams);
  
  cv::Mat mat = cv::imread("<Image_file>");
  
  auto ranklist = ir::IrInstance::retrieve(mat);
  
  return 0;
}
