#include "gtest/gtest.h"

#include <boost/filesystem.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <utils/hesaff/hesaff.h>


TEST(TestHesaff, TestHesaff_ouput) {
  auto sourceDir = boost::filesystem::path(__FILE__).parent_path();
  auto imageFile = sourceDir /
                   boost::filesystem::path("data") /
                   boost::filesystem::path("park.jpg");
  auto siftFile = sourceDir /
                  boost::filesystem::path("data") /
                  boost::filesystem::path("park.sift");

  // Get output from hesaff
  auto img = cv::imread(imageFile.string());
  boost::multi_array<double, 2> kps, descs;
  hesaff::extract(img, kps, descs);

  DLOG(INFO) << "Keypoint shape: " << kps.shape();
  DLOG(INFO) << "Descriptor shape: " << descs.shape();

  // Get output from groundtruth
  std::ifstream ifs(siftFile.string());
  size_t nDims, nFeats;
  ifs >> nDims >> nFeats;

  boost::multi_array<double, 2> expKps(boost::extents[nFeats][5]);
  boost::multi_array<double, 2> expDescs(boost::extents[nFeats][nDims]);
  for (size_t i = 0; i < nFeats; ++i) {
    float u, v, a, b, c;
    ifs >> u >> v >> a >> b >> c;
    expKps[i][0] = u;
    expKps[i][1] = v;
    expKps[i][2] = a;
    expKps[i][3] = b;
    expKps[i][4] = c;

    for (size_t j = 0; j < nDims; ++j) {
      float x;
      ifs >> x;
      expDescs[i][j] = x;
    }
  }

  ifs.close();

  // Compare
  ASSERT_EQ(kps.shape()[0], expKps.shape()[0]);
  ASSERT_EQ(kps.shape()[1], expKps.shape()[1]);
  ASSERT_EQ(descs.shape()[0], expDescs.shape()[0]);
  ASSERT_EQ(descs.shape()[1], expDescs.shape()[1]);
  
  for (size_t i = 0; i < descs.shape()[0]; ++i) {
    for (size_t j = 0; j < descs.shape()[1]; ++j) {
      ASSERT_NEAR(descs[i][j], expDescs[i][j], 1e-2);
    }
  }
  for (size_t i = 0; i < kps.shape()[0]; ++i) {
    for (size_t j = 0; j < kps.shape()[1]; ++j) {
      ASSERT_NEAR(kps[i][j], expKps[i][j], 1e-2);
    }
  }
}
