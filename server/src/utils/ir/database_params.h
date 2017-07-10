//
//  database_params.h
//  server
//
//  Created by Bao Truong on 7/3/17.
//
//

#ifndef database_params_h
#define database_params_h

#include <boost/filesystem.hpp>
#include <string>
#include <vector>


namespace ir {

enum CacheTag {
  IMAGE,
  KEYPOINT,
  DESCRIPTOR,
  WEIGHT,
  INDEX,
  TERM_FREQUENCY
};

/**
 * A parameter type used for bag-of-words constructing.
 */
struct DatabaseParams {
  size_t nWords;
  size_t batchSize;
  std::string imageFolder;
  std::string cacheFolder;
  std::string dbFolder;

  DatabaseParams(
    size_t nWords = 1000000,
    size_t batchSize = 100,
    std::string imageFolder = "",
    std::string cacheFolder = "",
    std::string dbFolder = ""):
    nWords(nWords),
    batchSize(batchSize),
    imageFolder(imageFolder),
    cacheFolder(cacheFolder),
    dbFolder(dbFolder) {

  }

  /**
   * Returns a list of filenames of images.
   * The returned results are just filenames, use `getFullPath` to get the
   * full path of each file.
   */
  std::vector<std::string> getDocuments();

  /**
   * Returns the full path of an image associated with a `CacheTag`.
   */
  std::string getFullPath(const std::string& docName, CacheTag tag = IMAGE);

  /**
   * Returns the path of a batch of the database.
   */
  std::string getDatabasePath(const size_t &batchId);
};

}

#endif /* database_params_h */
