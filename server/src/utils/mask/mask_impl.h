#ifndef UTILS_MASK_IMPL_H
#define UTILS_MASK_IMPL_H

#include "mask_interface.h"

#include <opencv2/opencv.hpp>


/**
 * An implementation of MaskInterface.
 *
 * This implementation eliminates static background using background subtraction
 * and removes skin using skin thresholding detection.
 */
class MaskImpl : public MaskInterface {
 private:
   cv::Ptr<cv::BackgroundSubtractor> bgrSubtractor_;

 public:
  /**
   * Default constructor.
   */
  MaskImpl(): bgrSubtractor_(cv::createBackgroundSubtractorMOG2()) {}

  /**
   * Returns a mask of the current frame based on previous ones.
   */
  cv::Mat getMask(const cv::Mat& src) override;
};

#endif // UTILS_MASK_IMPL_H
