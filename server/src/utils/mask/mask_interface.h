
#ifndef mask_interface_h
#define mask_interface_h

#include <opencv2/opencv.hpp>


/**
 * An interface for masking continuous frames.
 */
class MaskInterface {
 public:
  /**
   * Returns a mask of the current frame based on previous ones.
   */
  virtual cv::Mat getMask(const cv::Mat& src) = 0;
};

#endif /* mask_interface_h */
