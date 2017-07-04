#include "mask_impl.h"

using std::min;
using std::max;

bool checkRGB(int r, int g, int b) {
  bool e1 =
    (r>95) && (g>40) && (b>20) &&
    ((max(r,max(g, b)) - min(r, min(g, b)))>15) && (abs(r-g)>15) && (r>g) && (r>b);
  bool e2 = (r>220) && (g>210) && (b>170) && (abs(r-g)<=15) && (r>b) && (g>b);
  return e1 || e2;
}

bool checkYCrCb(float y, float cr, float cb) {
  bool e3 = cr <= 1.5862*cb+20;
  bool e4 = cr >= 0.3448*cb+76.2069;
  bool e5 = cr >= -4.5652*cb+234.5652;
  bool e6 = cr <= -1.15*cb+301.75;
  bool e7 = cr <= -2.2857*cb+432.85;
  return e3 && e4 && e5 && e6 && e7;
}

bool checkHSV(float h, float s, float v) {
  return (h < 25) || (h > 230);
}

cv::Mat getSkinMask(const cv::Mat &src) {
  cv::Mat dst(src.rows, src.cols, CV_8UC1);

  cv::Mat srcYCrCb, srcHSV;
  cv::cvtColor(src, srcYCrCb, CV_BGR2YCrCb);
  src.convertTo(srcHSV, CV_32FC3);
  cv::cvtColor(src, srcHSV, CV_BGR2HSV);
  cv::normalize(srcHSV, srcHSV, 0, 255, cv::NORM_MINMAX, CV_32FC3);

  for (int i = 0; i < src.rows; ++i) {
    for (int j = 0; j < src.cols; ++j) {
      bool ok = true;

      auto bgr = src.ptr<cv::Vec3b>(i)[j];
      ok &= checkRGB(bgr.val[2], bgr.val[1], bgr.val[0]);

      auto ycrcb = srcYCrCb.ptr<cv::Vec3b>(i)[j];
      ok &= checkYCrCb(ycrcb.val[0], ycrcb.val[1], ycrcb.val[2]);

      auto hsv = srcHSV.ptr<cv::Vec3b>(i)[j];
      ok &= checkHSV(hsv.val[0], hsv.val[1], hsv.val[2]);

      if (ok) {
        dst.ptr<uint8_t>(i)[j] = 0;
      } else {
        dst.ptr<uint8_t>(i)[j] = 255;
      }
    }
  }
  return dst;
}

cv::Mat MaskImpl::getMask(const cv::Mat& src) {
  // Remove background
  cv::Mat fgMask;
  this->bgrSubtractor_->apply(src, fgMask);
  // Remove skin
  cv::Mat skinMask = getSkinMask(src);

  return min(fgMask, skinMask);
}
