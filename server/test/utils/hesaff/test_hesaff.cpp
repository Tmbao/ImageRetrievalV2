#include "gtest/gtest.h"

#include <boost/filesystem.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <utils/hesaff/hesaff.h>


TEST(testHesaff, testHesaff_ouput) {
  auto sourceDir = boost::filesystem::path(__FILE__).parent_path();
  auto imageFile = sourceDir /
    boost::filesystem::path("data") /
    boost::filesystem::path("park.jpg");
  auto siftFile = sourceDir /
    boost::filesystem::path("data") /
    boost::filesystem::path("park.sift");
  
  // Get output from hesaff
  auto img = cv::imread(imageFile.string());
  af::array kps, descs;
  hesaff::extract(img, kps, descs);
  
  // Get output from groundtruth
  std::ifstream ifs(siftFile.string());
  size_t nDims, nFeats;
  ifs >> nDims >> nFeats;
  
  auto expKps = af::constant(0, nFeats, 5);
  auto expDescs = af::constant(0, nFeats, nDims);
  for (size_t i = 0; i < nFeats; ++i) {
    float u, v, a, b, c;
    ifs >> u >> v >> a >> b >> c;
    expKps(i, 0) = u;
    expKps(i, 1) = v;
    expKps(i, 2) = a;
    expKps(i, 3) = b;
    expKps(i, 4) = c;
    
    for (size_t j = 0; j < nDims; ++j) {
      float x;
      ifs >> x;
      expDescs(i, j) = x;
    }
  }
  
  ifs.close();
  
  // Compare
  ASSERT_EQ(kps.dims(), expKps.dims());
  ASSERT_EQ(descs.dims(), expDescs.dims());

  //TODO: Assert_eq elements of these arrays
//  float* kpData = kps.host<float>();
//  float* expKpData = expKps.host<float>();
//  float* descData = descs.host<float>();
//  float* expDescData = expDescs.host<float>();
}
