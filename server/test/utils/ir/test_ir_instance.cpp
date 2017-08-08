#include "gtest/gtest.h"

#include <boost/filesystem.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <fstream>
#include <glog/logging.h>
#include <utils/ir/ir_instance.h>


class TestIRInstance : public ::testing::Test {
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

    GlobalParams globalParams(false, 128);
    ir::QuantizationParams quantParams(
      8,
      3,
      800,
      6250,
      codebookFile.string(),
      "clusters",
      indexFile.string());
    ir::DatabaseParams dbParams(1000000, imageFolder.string(), cacheFolder.string());

    ir::IrInstance::createInstanceIfNecessary(
      globalParams,
      quantParams,
      dbParams);
  }
};

inline std::string getNameWithoutExtension(const std::string &filename) {
  return filename.substr(0, filename.rfind("."));
}

inline std::set<std::string> getGroundtruth(
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

inline float computeAP(
  const std::set<std::string> &pos,
  const std::set<std::string> &amb,
  const std::vector<ir::IrResult> &ranklist) {
  float oldRecall = 0;
  float oldPrecision = 1;
  float ap = 0;

  size_t intersectSize = 0;
  for (size_t i = 0, j = 0; i < ranklist.size(); ++i) {
    if (amb.count(getNameWithoutExtension(ranklist.at(i).name()))) {
      continue;
    }
    if (pos.count(getNameWithoutExtension(ranklist.at(i).name()))) {
      ++intersectSize;
    } else {
      DLOG(INFO) << getNameWithoutExtension(ranklist.at(i).name());
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

inline void saveRanklist(
  const boost::filesystem::path &ranklistFolder,
  const std::string &queryName,
  const std::vector<ir::IrResult> &ranklist,
  const std::set<std::string> &pos,
  const std::set<std::string> &amb,
  double ap) {
  std::string ranklistPath = (ranklistFolder / boost::filesystem::path(queryName)).string();
  std::ofstream ofs(ranklistPath);
  for (auto &item : ranklist) {
    if (pos.count(getNameWithoutExtension(item.name()))) {
      ofs << "POS_";
    } else {
      ofs << "NEG_";
    }
    ofs << getNameWithoutExtension(item.name()) << std::endl;
  }
  ofs << ap;
  ofs.close();
}

TEST_F(TestIRInstance, TestIrInstance_init) {
  // This function is intentionally left blank.
}

TEST_F(TestIRInstance, TestIrInstance_map) {
  auto sourceDir = boost::filesystem::path(__FILE__).parent_path();
  auto queryFolder = sourceDir /
                     boost::filesystem::path("data") /
                     boost::filesystem::path("query");
  auto gtFolder = sourceDir /
                  boost::filesystem::path("data") /
                  boost::filesystem::path("groundtruth");
  auto ranklistFolder = sourceDir /
                        boost::filesystem::path("data") /
                        boost::filesystem::path("ranklists");

  ir::DatabaseParams queryParams(1000000, queryFolder.string());

  auto queries = queryParams.getDocuments();
  float map = 0;
  for (auto &query : queries) {
    DLOG(INFO) << "Started processing " << query;
    auto fullPath = queryParams.getFullPath(query);
    auto queryMat = cv::imread(fullPath);
    auto ranklist = ir::IrInstance::retrieve(queryMat);

    // Verify ranklist
    for (auto &item : ranklist) {
      ASSERT_FALSE(boost::math::isnan(item.score()));
    }

    auto goodSet = getGroundtruth(gtFolder, query, "good");
    auto okSet = getGroundtruth(gtFolder, query, "ok");
    auto junkSet = getGroundtruth(gtFolder, query, "junk");

    goodSet.insert(okSet.begin(), okSet.end());

    auto ap = computeAP(goodSet, junkSet, ranklist);
    map += ap;

    DLOG(INFO) << "Finished processing " << query << ", AP = " << ap;
  }
  map /= queries.size();

  LOG(INFO) << "MAP = " << map;
  EXPECT_GT(map, 0.82);
}

TEST_F(TestIRInstance, TestIrInstance_map_parallel) {
  auto sourceDir = boost::filesystem::path(__FILE__).parent_path();
  auto queryFolder = sourceDir /
                     boost::filesystem::path("data") /
                     boost::filesystem::path("query");
  auto gtFolder = sourceDir /
                  boost::filesystem::path("data") /
                  boost::filesystem::path("groundtruth");
  auto ranklistFolder = sourceDir /
                        boost::filesystem::path("data") /
                        boost::filesystem::path("ranklists");

  ir::DatabaseParams queryParams(1000000, queryFolder.string());

  auto queries = queryParams.getDocuments();

  std::vector<cv::Mat> queryMats;
  for (auto &query : queries) {
    auto fullPath = queryParams.getFullPath(query);
    queryMats.push_back(cv::imread(fullPath));
  }

  auto ranklists = ir::IrInstance::retrieve(queryMats);

  float map = 0;
  for (size_t i = 0; i < ranklists.size(); ++i) {
    auto &ranklist = ranklists.at(i);

    // Verify ranklist
    for (auto &item : ranklist) {
      ASSERT_FALSE(boost::math::isnan(item.score()));
    }

    auto goodSet = getGroundtruth(gtFolder, queries.at(i), "good");
    auto okSet = getGroundtruth(gtFolder, queries.at(i), "ok");
    auto junkSet = getGroundtruth(gtFolder, queries.at(i), "junk");

    goodSet.insert(okSet.begin(), okSet.end());

    auto ap = computeAP(goodSet, junkSet, ranklist);
    map += ap;

    DLOG(INFO) << "Finished evaluating " << queries.at(i) << ", AP = " << ap;
//    saveRanklist(ranklistFolder, queries.at(i), ranklist, goodSet, junkSet, ap);
  }
  map /= queries.size();

  LOG(INFO) << "MAP = " << map;
  EXPECT_GT(map, 0.82);
}
