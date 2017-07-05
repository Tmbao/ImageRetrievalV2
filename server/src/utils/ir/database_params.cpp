//
//  database_params.cpp
//  server
//
//  Created by Bao Truong on 7/3/17.
//
//

#include "database_params.h"

#include <boost/range/iterator_range.hpp>


boost::container::vector<std::string> ir::DatabaseParams::getDocuments() {
  boost::container::vector<std::string> docs;
  for (auto& entry :
       boost::make_iterator_range(
         boost::filesystem::directory_iterator(imageFolder),
         {})) {
    docs.push_back(entry.path().filename().string());

  }
  return docs;
}

std::string joinPath(const std::string& dir, const std::string& file) {
  return (boost::filesystem::path(dir) / boost::filesystem::path(file)).string();
}

std::string ir::DatabaseParams::getFullPath(const std::string& docName, CacheTag tag) {
  switch (tag) {
  case IMAGE:
    return joinPath(imageFolder, docName);
  case KEYPOINT:
    return joinPath(cacheFolder, docName + ".kp");
  case DESCRIPTOR:
    return joinPath(cacheFolder, docName + ".des");
  case WEIGHT:
    return joinPath(cacheFolder, docName + ".wt");
  case INDEX:
    return joinPath(cacheFolder, docName + ".id");
  case TERM_FREQUENCY:
    return joinPath(cacheFolder, docName + ".tf");
  default:
    throw "Tag not found!";
  }
}
