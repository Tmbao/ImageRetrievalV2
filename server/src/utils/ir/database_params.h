//
//  database_params.h
//  server
//
//  Created by Bao Truong on 7/3/17.
//
//

#ifndef database_params_h
#define database_params_h

#include <boost/container/vector.hpp>
#include <boost/filesystem.hpp>
#include <string>


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
  int nWords;
  std::string imageFolder;
  std::string cacheFolder;

  DatabaseParams() {
    nWords = 1000000;
    //TODO: Update these
    imageFolder = "";
    cacheFolder = "";

    // Create cache directory if necessary
    /*
    if (!boost::filesystem::exists(boost::filesystem::path(cacheFolder))) {
      boost::filesystem::create_directory(boost::filesystem::path(cacheFolder));
    }
    */
  }

  /**
   * Returns a list of filenames of images.
   * The returned results are just filenames, use `getFullPath` to get the
   * full path of each file.
   */
  boost::container::vector<std::string> getDocuments();

  /**
   * Get the full path of an image associated with a `CacheTag`.
   */
  std::string getFullPath(const std::string& docName, CacheTag tag = IMAGE);
};

}

#endif /* database_params_h */
