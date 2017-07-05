//
//  ir_instance.cpp
//  server
//
//  Created by Bao Truong on 7/3/17.
//
//

#include "ir_instance.h"

#include <algorithm>
#include <boost/asio.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/thread.hpp>
#include <flann/io/hdf5.h>
#include <fstream>
#include <numeric>

#include "../hesaff/hesaff.h"


ir::IrInstance::IrInstance() {
  // Impelementation goes here!
  buildIndexIfNecessary();
  buildDatabase();
}

void ir::IrInstance::createInstanceIfNecessary() {
  boost::mutex::scoped_lock scopedLock(initMutex_);
  if (instance_ == NULL) {
    instance_ = new ir::IrInstance();
  }
}

void ir::IrInstance::buildIndexIfNecessary() {
  flann::Matrix<float> codebook;
  flann::load_from_file(
    codebook,
    quantParams_.codebookFile,
    quantParams_.codebookName);

  flann::IndexParams* indexParams;
  if (!globalParams.overwrite && boost::filesystem::exists(quantParams_.indexFile)) {
    indexParams = new flann::SavedIndexParams(quantParams_.indexFile);
  } else {
    indexParams = new flann::KDTreeIndexParams(quantParams_.nTrees);
  }

  quantIndex_ = new flann::Index< flann::L2<float> >(codebook, *indexParams);
  quantIndex_->buildIndex();
  quantIndex_->save(quantParams_.indexFile);
}

void ir::IrInstance::extractDbIfNecessary(
  const std::string &docName,
  af::array &keypoints,
  af::array &descriptors) {

  std::string imagePath = dbParams_.getFullPath(docName, IMAGE);
  std::string descPath = dbParams_.getFullPath(docName, DESCRIPTOR);
  std::string kpPath = dbParams_.getFullPath(docName, KEYPOINT);

  // Check if the feature exists
  if (boost::filesystem::exists(descPath) &&
      boost::filesystem::exists(kpPath) &&
      !globalParams.overwrite) {
    keypoints = af::readArray(kpPath.c_str(), "");
    descriptors = af::readArray(descPath.c_str(), "");
  } else {
    cv::Mat image = cv::imread(imagePath);
    af::array keypoints, descriptors;
    hesaff::extract(image, keypoints, descriptors);
    af::saveArray("", keypoints, kpPath.c_str());
    af::saveArray("", descriptors, kpPath.c_str());
  }
}

template <typename T>
void saveVector(const std::vector<T> &data, const std::string &filename) {
  std::ofstream ofs(filename);
  boost::archive::binary_oarchive bo(ofs);
  bo << data;
}

template <typename T>
void loadVector(std::vector<T> &data, const std::string &filename) {
  std::ifstream ifs(filename);
  boost::archive::binary_iarchive bi(ifs);
  bi >> data;
}

void ir::IrInstance::quantizeDbIfNecessary(
  const std::string &docName,
  const af::array &descriptors,
  std::vector<size_t> &indices,
  std::vector<float> &weights) {

  std::string indexPath = dbParams_.getFullPath(docName, INDEX);
  std::string weightPath = dbParams_.getFullPath(docName, WEIGHT);

  // Check if quantization data exists
  if (boost::filesystem::exists(indexPath) &&
      boost::filesystem::exists(weightPath) &&
      !globalParams.overwrite) {
    loadVector(indices, indexPath);
    loadVector(weights, weightPath);
  } else {
    quantize(descriptors, indices, weights);
    saveVector(indices, indexPath);
    saveVector(weights, weightPath);
  }
}

void ir::IrInstance::computeTFDbIfNecessary(
  const std::string &docName,
  const std::vector<size_t> &indices,
  const std::vector<float> &weights,
  af::array &termFreq) {

  std::string tfPath = dbParams_.getFullPath(docName, TERM_FREQUENCY);

  // Check if tf exists
  if (boost::filesystem::exists(tfPath) && !globalParams.overwrite) {
    termFreq = af::readArray(tfPath.c_str(), "");
  } else {
    computeTF(indices, weights, termFreq);
    af::saveArray("", termFreq, tfPath.c_str());
  }
}

void ir::IrInstance::loadDocumentTask(
  ir::IrInstance* &instance,
  boost::container::vector< boost::container::set<size_t> > &rawInvIndex,
  const size_t &docId) {

  std::string docName = instance->docNames_.at(docId);

  // Extract features
  af::array keypoints, descriptors;
  instance->extractDbIfNecessary(docName, keypoints, descriptors);

  // Quantize
  std::vector<size_t> indices;
  std::vector<float> weights;
  instance->quantizeDbIfNecessary(docName, descriptors, indices, weights);

  // Build bag-of-words vector
  af::array termFreq;
  instance->computeTFDbIfNecessary(docName, indices, weights, termFreq);

  // Update database
  instance->database_(docId) = termFreq;

  // Add indices to inverted index and update idf
  for (size_t i = 0; i < indices.size(); ++i) {
    rawInvIndex.at(indices.at(i)).insert(docId);
    instance->invDocFreq_(indices.at(i)) += 1;
  }
}

void ir::IrInstance::buildDatabase() {
  docNames_ = dbParams_.getDocuments();
  size_t nDocs = docNames_.size();

  /**
   The following stuffs will be done sequentially and in parallel
   - Extract features (hesaff)
   - Quantize (quantize)
   - Compute tf (computeTF)
   - Add features to inverted index (invIndex_)
   - Compute idf (invDocFreq_)
   */

  // Initialize database
  database_ = af::sparse(af::array(nDocs, dbParams_.nWords));
  boost::container::vector< boost::container::set<size_t> > rawInvIndex(dbParams_.nWords);
  invDocFreq_ = af::constant(0, dbParams_.nWords);

  // Initialize a thread pool
  boost::thread_group threads;
  boost::asio::io_service ioService;
  for (size_t i = 0; i < globalParams.nThreads; ++i) {
    threads.create_thread(boost::bind(&boost::asio::io_service::run, &ioService));
  }

  // Initialize tasks
  for (size_t i = 0; i < docNames_.size(); ++i) {
    ioService.post(boost::bind(loadDocumentTask, this, rawInvIndex, i));
  }

  // Join tasks
  ioService.stop();
  threads.join_all();

  // Compute idf
  invDocFreq_ = af::log((float) nDocs / (1 + invDocFreq_));

  // Build inverted index
  for (size_t i = 0; i < rawInvIndex.size(); ++i) {
    size_t *indexData = new size_t[rawInvIndex.at(i).size()];
    std::copy(rawInvIndex.at(i).begin(), rawInvIndex.at(i).end(), indexData);

    invIndex_.at(i) = af::array(rawInvIndex.at(i).size(), indexData);
  }
}

void ir::IrInstance::quantize(
  const af::array &descriptors,
  std::vector<size_t> &termIndices,
  std::vector<float> &termWeights) {

  flann::Matrix<float> queries(
    descriptors.host<float>(),
    descriptors.dims(0),
    descriptors.dims(1));

  flann::Matrix<int> indices(
    new int[descriptors.dims(0) * quantParams_.knn],
    descriptors.dims(0),
    quantParams_.knn);

  flann::Matrix<float> dists(
    new float[descriptors.dims(0) * quantParams_.knn],
    descriptors.dims(0),
    quantParams_.knn);

  quantIndex_->knnSearch(
    queries,
    indices,
    dists,
    quantParams_.knn,
    flann::SearchParams(quantParams_.nChecks));

  // Fetch indices
  af::array afIndices(indices.rows, indices.cols, indices.ptr());

  // Fetch weights, apply radial weighting and normalize
  af::array afWeights(dists.rows, dists.cols, dists.ptr());
  afWeights = af::exp(-afWeights / (2 * quantParams_.deltaSqr));
  afWeights = afWeights / af::tile(af::sum(afWeights, 1), 1, afWeights.dims(1));

  // Flatten
  afIndices = af::flat(afIndices);
  afWeights = af::flat(afWeights);

  // Copy back to vector
  int* ptrIndices = afIndices.host<int>();
  float* ptrWeights = afWeights.host<float>();
  termIndices = std::vector<size_t>(ptrIndices, ptrIndices + afIndices.dims(0));
  termWeights = std::vector<float>(ptrWeights, ptrWeights + afWeights.dims(0));
}

void ir::IrInstance::computeTF(
  const std::vector<size_t> &indices,
  const std::vector<float> &weights,
  af::array &termFreq) {

  // Join indices and weights
  termFreq = af::constant(0, dbParams_.nWords);
  af::array rawFreq = af::constant(0, dbParams_.nWords);
  for (size_t i = 0; i < indices.size(); ++i) {
    termFreq(indices.at(i)) += weights.at(i);
    rawFreq(indices.at(i)) += 1;
  }
  termFreq = sparse(termFreq);
  rawFreq = sparse(rawFreq);

  // Compute tfidf
  termFreq /= af::sqrt(af::abs(rawFreq));
  float totalFreq = *af::sum(termFreq).host<float>();
  termFreq /= af::sqrt(af::abs(termFreq / totalFreq));
}

void ir::IrInstance::computeScore(const af::array &bow, boost::container::vector<float> &scores) {
  af::array afScores = af::matmul(database_, bow);
  float* ptrScores = afScores.host<float>();
  scores = boost::container::vector<float>(ptrScores, ptrScores + afScores.dims(0));
}

boost::container::vector<ir::IrResult> ir::IrInstance::retrieve(const cv::Mat &image, int topK) {
  createInstanceIfNecessary();

  // Extract features from the image using perdoch's hessaff
  af::array keypoints, descriptors;
  hesaff::extract(image.clone(), keypoints, descriptors);

  // Carry out quantization, using soft-assignment with `quantParams_`
  std::vector<size_t> indices;
  std::vector<float> weights;
  quantize(descriptors, indices, weights);

  // Build bag-of-words vector
  af::array bow;
  computeTF(indices, weights, bow);
  // Compute tfidf
  // TODO: Check sparse * sparse == sparse ?
  bow *= invDocFreq_;

  // Compute scores
  boost::container::vector<float> scores;
  computeScore(bow, scores);

  // Sort scores and return
  boost::container::vector<size_t> docIndices;
  std::iota(docIndices.begin(), docIndices.end(), 0);

  std::sort(
    docIndices.begin(),
    docIndices.end(),
  [&scores](const size_t &lhs, const size_t &rhs) -> int {
    if (scores.at(lhs) > scores.at(rhs)) {
      return -1;
    } else if (scores.at(lhs) < scores.at(rhs)) {
      return 1;
    } else {
      return 0;
    }
  });

  boost::container::vector<ir::IrResult> result;
  if (topK == -1) {
    for (size_t id : docIndices) {
      result.push_back(ir::IrResult(docNames_.at(id), scores.at(id)));
    }
  } else {
    for (size_t i = 0; i < topK; ++i) {
      size_t id = docIndices.at(i);
      result.push_back(ir::IrResult(docNames_.at(id), scores.at(id)));
    }
  }
  return result;
}
