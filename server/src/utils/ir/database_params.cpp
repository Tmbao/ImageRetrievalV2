//
//  database_params.cpp
//  server
//
//  Created by Bao Truong on 7/3/17.
//
//

#include "database_params.h"

#include <boost/range/iterator_range.hpp>


std::vector<std::string> DatabaseParams::getDocuments() {
  std::vector<std::string> docs;
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

std::string DatabaseParams::getFullPath(const std::string& docName, CacheTag tag) {
  switch (tag) {
  case IMAGE:
    return joinPath(imageFolder, docName);
  case KEYPOINT:
    return joinPath(cacheFolder, docName + ".keypoint");
  case DESCRIPTOR:
    return joinPath(cacheFolder, docName + ".descriptor");
  case WEIGHT:
    return joinPath(cacheFolder, docName + ".weight");
  case INDEX:
    return joinPath(cacheFolder, docName + ".index");
  default:
    throw "Tag not found!";
  }
}
