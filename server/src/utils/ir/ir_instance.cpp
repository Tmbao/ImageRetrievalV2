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
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>
#include <flann/io/hdf5.h>
#include <numeric>

#include "../hesaff/hesaff.h"


IrInstance::IrInstance() {
  // Impelementation goes here!
  buildIndexIfNecessary();
  buildDatabase();
}

void IrInstance::createInstanceIfNecessary() {
  boost::mutex::scoped_lock scopedLock(initMutex_);
  if (instance_ == NULL) {
    instance_ = new IrInstance();
  }
}

void IrInstance::buildIndexIfNecessary(bool overwrite) {
  flann::Matrix<float> codebook;
  flann::load_from_file(
    codebook,
    quantParams_.codebookFile,
    quantParams_.codebookName);

  flann::IndexParams* indexParams;
  if (!overwrite && boost::filesystem::exists(quantParams_.indexFile)) {
    indexParams = new flann::SavedIndexParams(quantParams_.indexFile);
  } else {
    indexParams = new flann::KDTreeIndexParams(quantParams_.nTrees);
  }

  quantIndex_ = new flann::Index< flann::L2<float> >(codebook, *indexParams);
  quantIndex_->buildIndex();
  quantIndex_->save(quantParams_.indexFile);
}

void extractDbIfNecessary(
  const std::string &docName,
  af::array &keypoints,
  af::array &descriptors,
  bool overwrite) {

}

void quantizeDbIfNecessary(
  const std::string &docName,
  const af::array &descriptors,
  std::vector<size_t> &indices,
  std::vector<float> &weights,
  bool overwrite) {

}

void computeTFIfNecessary(
                          const std::string &docName,
                          const std::vector<size_t> &indices,
                          const std::vector<float> &weights,
                          af::array &termFreq,
                          bool overwrite) {
  
}

void IrInstance::buildDatabase(bool overwrite, size_t nThreads) {
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
  database_ = af::sparse(af::constant(0, nDocs, dbParams_.nWords));

  // Initialize a thread pool
  boost::thread_group threads;
  boost::asio::io_service ioService;
  for (size_t i = 0; i < nThreads; ++i) {
    threads.create_thread(boost::bind(&boost::asio::io_service::run, &ioService));
  }

  // Initialize tasks
  for (size_t i = 0; i < docNames_.size(); ++i) {
    ioService.post(boost::bind(
    [&](const size_t &docId, bool overwrite) {
      std::string docName = docNames_.at(docId);

      // Extract features
      af::array keypoints, descriptors;
      extractDbIfNecessary(docName, keypoints, descriptors, overwrite);

      // Quantize
      std::vector<size_t> indices;
      std::vector<float> weights;
      quantizeDbIfNecessary(docName, descriptors, indices, weights, overwrite);
      
      // Build bag-of-words vector
      af::array termFreq;
      computeTFIfNecessary(docName, indices, weights, termFreq, overwrite);
      
      database_(docId) = termFreq;
    },
    i, overwrite));
  }
  
  // Join tasks
  ioService.stop();
  threads.join_all();
  
  
}

void IrInstance::quantize(
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
  afWeights =
    afWeights / af::tile(af::sum(afWeights, 1), 1, afWeights.dims(1));

  // Flatten
  afIndices = af::flat(afIndices);
  afWeights = af::flat(afWeights);

  // Copy back to vector
  int* ptrIndices = afIndices.host<int>();
  float* ptrWeights = afWeights.host<float>();
  termIndices = std::vector<size_t>(ptrIndices, ptrIndices + afIndices.dims(0));
  termWeights = std::vector<float>(ptrWeights, ptrWeights + afWeights.dims(0));
}

void IrInstance::computeTF(
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

void IrInstance::computeScore(const af::array &bow, std::vector<float> &scores) {
  af::array afScores = af::matmul(database_, bow);
  float* ptrScores = afScores.host<float>();
  scores = std::vector<float>(ptrScores, ptrScores + afScores.dims(0));
}

std::vector<IrResult> IrInstance::retrieve(const cv::Mat &image, int topK) {
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
  std::vector<float> scores;
  computeScore(bow, scores);

  // Sort scores and return
  std::vector<size_t> docIndices;
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

  std::vector<IrResult> result;
  if (topK == -1) {
    for (size_t id : docIndices) {
      result.push_back(IrResult(docNames_.at(id), scores.at(id)));
    }
  } else {
    for (size_t i = 0; i < topK; ++i) {
      size_t id = docIndices.at(i);
      result.push_back(IrResult(docNames_.at(id), scores.at(id)));
    }
  }
  return result;
}
