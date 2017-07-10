#include "gtest/gtest.h"

#include <boost/filesystem.hpp>
#include <fstream>
#include <glog/logging.h>
#include <utils/ir/ir_instance.h>

class TestIR : public ::testing::Test {
 protected:
  virtual void SetUp() {
    auto sourceDir = boost::filesystem::path(__FILE__).parent_path();
    auto imageFolder = sourceDir /
                       boost::filesystem::path("data") /
                       boost::filesystem::path("image");
    auto cacheFolder = sourceDir /
                       boost::filesystem::path("data") /
                       boost::filesystem::path("cache");
    auto codebookFile = sourceDir /
                        boost::filesystem::path("data") /
                        boost::filesystem::path("Clustering_l2_1000000_13516675_128_50it.hdf5");
    auto indexFile = sourceDir /
                     boost::filesystem::path("data") /
                     boost::filesystem::path("index.hdf5");

    GlobalParams globalParams(false);
    ir::QuantizationParams quantParams(
      8,
      3,
      800,
      6250,
      codebookFile.string(),
      "clusters",
      indexFile.string());
    ir::DatabaseParams dbParams(1000000, 128, imageFolder.string(), cacheFolder.string());

    ir::IrInstance::createInstanceIfNecessary(globalParams, quantParams, dbParams);
  }
};

std::string getNameWithoutExtension(std::string filename) {
  return filename.substr(0, filename.rfind("."));
}

std::set<std::string> getGroundtruth(
  const boost::filesystem::path &gtFolder,
  const std::string &queryName,
  const std::string &tag) {
  std::string gtName = getNameWithoutExtension(queryName) + "_" + tag + ".txt";
  std::string gtPath = (gtFolder / boost::filesystem::path(gtName)).string();

  std::set<std::string> lst;
  std::ifstream ifs(gtPath);
  for (std::string line; ifs >> line;) {
    lst.insert(line);
  }
  ifs.close();
  return lst;
}

float computeAP(
  const std::set<std::string> &pos,
  const std::set<std::string> &amb,
  const std::vector<ir::IrResult> &ranklist) {
  float oldRecall = 0;
  float oldPrecision = 1;
  float ap = 0;

  size_t intersectSize = 0;
  for (size_t i = 0, j = 0; i < ranklist.size(); ++i) {
    ASSERT_EQ(std::isnan(ranklist.at(i).scores(), false));

    if (amb.count(getNameWithoutExtension(ranklist.at(i).name()))) {
      continue;
    }
    if (pos.count(getNameWithoutExtension(ranklist.at(i).name()))) {
      ++intersectSize;
    }

    float recall = (float) intersectSize / pos.size();
    float precision = (float) intersectSize / (j + 1);

    ap += (recall - oldRecall) * ((oldPrecision + precision) / 2);

    oldRecall = recall;
    oldPrecision = precision;
    ++j;
  }
  return ap;
}

TEST_F(TestIR, TestIrInstance_map) {
  auto sourceDir = boost::filesystem::path(__FILE__).parent_path();
  auto queryFolder = sourceDir /
                     boost::filesystem::path("data") /
                     boost::filesystem::path("query");
  auto gtFolder = sourceDir /
                  boost::filesystem::path("data") /
                  boost::filesystem::path("groundtruth");

  ir::DatabaseParams queryParams(1000000, 100, queryFolder.string());

  auto queries = queryParams.getDocuments();
  float map = 0;
  for (auto &query : queries) {
    DLOG(INFO) << "Started processing " << query;
    auto fullPath = queryParams.getFullPath(query);
    auto queryMat = cv::imread(fullPath);
    auto ranklist = ir::IrInstance::retrieve(queryMat);

    auto goodSet = getGroundtruth(gtFolder, query, "good");
    auto okSet = getGroundtruth(gtFolder, query, "ok");
    auto junkSet = getGroundtruth(gtFolder, query, "junk");

    goodSet.insert(okSet.begin(), okSet.end());

    ASSERT_EQ(query, ranklist.at(0).name());
    auto ap = computeAP(goodSet, junkSet, ranklist);
    map += ap;

    DLOG(INFO) << "Finished processing " << query << ", AP = " << ap;
  }
  map /= queries.size();

  DLOG(INFO) << "MAP = " << map;
  EXPECT_GT(map, 0.8);
}
