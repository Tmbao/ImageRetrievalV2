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
#include <boost/fusion/algorithm/transformation/flatten.hpp>
#include <boost/fusion/include/flatten.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/thread.hpp>
#include <boost/range/algorithm/for_each.hpp>
#include <flann/io/hdf5.h>
#include <fstream>
#include <glog/logging.h>
#include <numeric>

#include "../hesaff/hesaff.h"


std::shared_ptr<ir::IrInstance> ir::IrInstance::instance_ = nullptr;
boost::mutex ir::IrInstance::initMutex_;
GlobalParams ir::IrInstance::globalParams_;
ir::QuantizationParams ir::IrInstance::quantParams_;
ir::DatabaseParams ir::IrInstance::dbParams_;

ir::IrInstance::IrInstance() {
  af::setDevice(0);
  af::info();

  buildIndexIfNecessary();
  buildDatabase();
}

void ir::IrInstance::createInstanceIfNecessary() {
  boost::mutex::scoped_lock scopedLock(initMutex_);
  if (instance_ == nullptr) {
    instance_ = std::shared_ptr<ir::IrInstance>(new ir::IrInstance());
  }
}

void ir::IrInstance::createInstanceIfNecessary(
  GlobalParams globalParams,
  QuantizationParams quantParams,
  DatabaseParams dbParams
) {
  boost::mutex::scoped_lock scopedLock(initMutex_);
  if (instance_ == nullptr) {
    globalParams_ = globalParams;
    quantParams_ = quantParams;
    dbParams_ = dbParams;
    instance_ = std::shared_ptr<ir::IrInstance>(new ir::IrInstance());
  }
}

template <typename T>
void save(const std::vector<T> &data, const std::string &filename) {
  std::ofstream ofs(filename);
  boost::archive::binary_oarchive bo(ofs);
  bo << data;
}

template <typename T>
void load(std::vector<T> &data, const std::string &filename) {
  std::ifstream ifs(filename);
  boost::archive::binary_iarchive bi(ifs);
  bi >> data;
}

template <typename T, size_t N>
void save(const boost::multi_array<T, N> &data, const std::string &filename) {
  std::ofstream ofs(filename);
  boost::archive::binary_oarchive bo(ofs);
  bo << boost::serialization::make_array(data.shape(), N);
  bo << boost::serialization::make_array(data.data(), data.num_elements());
}

template <typename T, size_t N>
void load(boost::multi_array<T, N> &data, const std::string &filename) {
  std::ifstream ifs(filename);
  boost::archive::binary_iarchive bi(ifs);
  boost::array<std::size_t, N> shape;
  bi >> boost::serialization::make_array(shape.data(), N);
  data.resize(shape);
  bi >> boost::serialization::make_array(data.data(), data.num_elements());
}

void ir::IrInstance::buildIndexIfNecessary() {
  DLOG(INFO) << "Started reading codebook file";
  flann::Matrix<float> codebook;
  flann::load_from_file(
    codebook,
    quantParams_.codebookFile,
    quantParams_.codebookName);
  DLOG(INFO) << "Finished reading codebook file";

  DLOG(INFO) << "Started creating index";
  flann::IndexParams* indexParams;
  if (!globalParams_.overwrite && boost::filesystem::exists(quantParams_.indexFile)) {
    indexParams = new flann::SavedIndexParams(quantParams_.indexFile);
  } else {
    indexParams = new flann::KDTreeIndexParams(quantParams_.nTrees);
  }
  DLOG(INFO) << "Finished creating index";

  DLOG(INFO) << "Started constructing KDTree";
  quantIndex_ = new flann::Index< flann::L2<float> >(codebook, *indexParams);
  quantIndex_->buildIndex();
  quantIndex_->save(quantParams_.indexFile);
  DLOG(INFO) << "Finished constructing KDTree";
}

void ir::IrInstance::extractDbIfNecessary(
  const std::string &docName,
  boost::multi_array<float, 2> &keypoints,
  boost::multi_array<float, 2> &descriptors) {

  std::string imagePath = dbParams_.getFullPath(docName, IMAGE);
  std::string descPath = dbParams_.getFullPath(docName, DESCRIPTOR);
  std::string kpPath = dbParams_.getFullPath(docName, KEYPOINT);

  // Check if the feature exists
  if (boost::filesystem::exists(descPath) &&
      boost::filesystem::exists(kpPath) &&
      !globalParams_.overwrite) {
    load(keypoints, kpPath);
    load(descriptors, descPath);
  } else {
    cv::Mat image = cv::imread(imagePath);
    hesaff::extract(image, keypoints, descriptors, true);
    save(keypoints, kpPath);
    save(descriptors, descPath);
  }
}

void ir::IrInstance::quantizeDbIfNecessary(
  const std::string &docName,
  boost::multi_array<float, 2> &descriptors,
  std::vector<size_t> &indices,
  std::vector<float> &weights) {

  std::string indexPath = dbParams_.getFullPath(docName, INDEX);
  std::string weightPath = dbParams_.getFullPath(docName, WEIGHT);

  // Check if quantization data exists
  if (boost::filesystem::exists(indexPath) &&
      boost::filesystem::exists(weightPath) &&
      !globalParams_.overwrite) {
    load(indices, indexPath);
    load(weights, weightPath);
  } else {
    quantize(descriptors, indices, weights);
    save(indices, indexPath);
    save(weights, weightPath);
  }
}

void ir::IrInstance::computeTFDbIfNecessary(
  const std::string &docName,
  const std::vector<size_t> &indices,
  const std::vector<float> &weights,
  af::array &termFreq) {

  std::string tfPath = dbParams_.getFullPath(docName, TERM_FREQUENCY);

  // Check if tf exists
  if (boost::filesystem::exists(tfPath) && !globalParams_.overwrite) {
    termFreq = af::readArray(tfPath.c_str(), "");
  } else {
    computeTF(indices, weights, termFreq);
    //TODO: Uncomment this
//    af::saveArray("", termFreq, tfPath.c_str());
  }
}

void ir::IrInstance::loadDocumentTask(
  IrInstance* &instance,
  const size_t &batchId,
  const size_t &docId,
  std::vector<float> *rawInvDocFreq,
  std::vector<boost::mutex> *rawInvMutex,
  std::vector< std::set<size_t> > *rawInvIndex) {

  std::string docName = instance->docNames_.at(docId);
  DLOG(INFO) << "Started loading document #" + std::to_string(docId) + " " + docName;

  // Make sure that all threads use a single device
  af::setDevice(0);

  // Extract features
  boost::multi_array<float, 2> keypoints, descriptors;
  instance->extractDbIfNecessary(docName, keypoints, descriptors);

  // Ignore documents with no keypoints
  if (keypoints.shape()[0] > 0) {
    // Quantize
    std::vector<size_t> indices;
    std::vector<float> weights;
    instance->quantizeDbIfNecessary(docName, descriptors, indices, weights);

    // Build bag-of-words vector
    af::array termFreq;
    instance->computeTFDbIfNecessary(docName, indices, weights, termFreq);

    // Update database
    instance->databaseMutex_.lock();
    instance->database_.at(batchId).row(docId - batchId * dbParams_.batchSize) = termFreq;
    instance->databaseMutex_.unlock();

    // Add indices to inverted index and update idf
    for (size_t i = 0; i < indices.size(); ++i) {
      rawInvMutex->at(indices.at(i)).lock();
      rawInvIndex->at(indices.at(i)).insert(docId);
      rawInvDocFreq->at(indices.at(i)) += 1;
      rawInvMutex->at(indices.at(i)).unlock();
    }
  }

  DLOG(INFO) << "Finished " + docName;
}

void ir::IrInstance::buildDatabaseOfBatchIfNecessary(
  const size_t &batchId,
  const size_t &fromDocId,
  const size_t &untilDocId,
  std::vector< std::set<size_t> > &rawInvIndex,
  std::vector<float> &rawInvDocFreq,
  std::vector<boost::mutex> &rawInvMutex) {

  /**
   The following stuffs will be done sequentially and in parallel
   - Extract features (hesaff)
   - Quantize (quantize)
   - Compute tf (computeTF)
   - Add features to inverted index (invIndex_)
   - Compute idf (invDocFreq_)
   */

  // Initialize database
  DLOG(INFO) << "Started intializing database #" << batchId;
  database_.at(batchId) = af::constant(0, untilDocId - fromDocId, dbParams_.nWords);

  // Initialize a thread pool
  boost::thread_group threads;
  boost::asio::io_service ioService;

  // Initialize tasks
  DLOG(INFO) << "Started adding tasks, nTasks = " << untilDocId - fromDocId;
  for (size_t docId = fromDocId; docId < untilDocId; ++docId) {
    ioService.post(boost::bind(
                     loadDocumentTask,
                     this,
                     batchId,
                     docId,
                     &rawInvDocFreq,
                     &rawInvMutex,
                     &rawInvIndex));
  }
  DLOG(INFO) << "Finished adding tasks, nTasks = " << untilDocId - fromDocId;

  // Initialize threads
  DLOG(INFO) << "Started adding threads, nThreads = " << globalParams_.nThreads;
  for (size_t i = 0; i < globalParams_.nThreads; ++i) {
    threads.create_thread(boost::bind(&boost::asio::io_service::run, &ioService));
  }
  DLOG(INFO) << "Finished adding threads, nThreads = " << globalParams_.nThreads;

  // Join tasks
  threads.join_all();
  ioService.stop();
  DLOG(INFO) << "Finished all tasks";

  DLOG(INFO) << "Started compressing database #" << batchId;
  database_.at(batchId) = af::sparse(database_.at(batchId));
  DLOG(INFO) << "Finished building database #" << batchId;
}

void ir::IrInstance::buildDatabase() {
  docNames_ = dbParams_.getDocuments();
  size_t nDocs = docNames_.size();

  DLOG(INFO) << "Number of documents " << nDocs;

  database_.resize((nDocs - 1) / dbParams_.batchSize + 1);
  std::vector< std::set<size_t> > rawInvIndex(dbParams_.nWords);
  std::vector<float> rawInvDocFreq(dbParams_.nWords);
  std::vector<boost::mutex> rawInvMutex(dbParams_.nWords);
  invIndex_.resize(dbParams_.nWords);

  for (size_t batchId = 0, fromDocId = 0;
       fromDocId < nDocs;
       fromDocId += dbParams_.batchSize, ++batchId) {
    size_t untilDocId = std::min(fromDocId + (size_t) dbParams_.batchSize, nDocs);
    buildDatabaseOfBatchIfNecessary(
      batchId,
      fromDocId,
      untilDocId,
      rawInvIndex,
      rawInvDocFreq,
      rawInvMutex);
  }

  // Compute idf
  DLOG(INFO) << "Started computing idf";
  for (size_t i = 0; i < rawInvDocFreq.size(); ++i) {
    if (rawInvDocFreq.at(i) > 0) {
      rawInvDocFreq.at(i) = log(nDocs / rawInvDocFreq.at(i));
    }
  }
  invDocFreq_ = af::array(rawInvDocFreq.size(), rawInvDocFreq.data());

  // Build inverted index (Temporarily disabled)
//  DLOG(INFO) << "Started building inverted index";
//  for (size_t i = 0; i < rawInvIndex.size(); ++i) {
//    if (rawInvIndex.at(i).size() == 0) {
//      continue;
//    }
//    size_t* indexData = new size_t[rawInvIndex.at(i).size()];
//    std::copy(rawInvIndex.at(i).begin(), rawInvIndex.at(i).end(), indexData);
//
//    invIndex_.at(i) = af::array(rawInvIndex.at(i).size(), indexData);
//  }
}

void ir::IrInstance::quantize(
  boost::multi_array<float, 2> &descriptors,
  std::vector<size_t> &termIndices,
  std::vector<float> &termWeights) {

  flann::Matrix<float> queries(
    descriptors.data(),
    descriptors.shape()[0],
    descriptors.shape()[1]);

  flann::Matrix<int> indices(
    new int[descriptors.shape()[0] * quantParams_.knn],
    descriptors.shape()[0],
    quantParams_.knn);

  flann::Matrix<float> dists(
    new float[descriptors.shape()[0] * quantParams_.knn],
    descriptors.shape()[0],
    quantParams_.knn);

  flann::SearchParams searchParams(quantParams_.nChecks);
  searchParams.cores = globalParams_.nThreads;

  quantIndex_->knnSearch(
    queries,
    indices,
    dists,
    quantParams_.knn,
    searchParams);

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
  std::vector<float> rawTermFreq(dbParams_.nWords, 0);
  std::vector<size_t> rawFreq(dbParams_.nWords, 0);
  std::set<size_t> uniqueIndices;
  for (size_t i = 0; i < indices.size(); ++i) {
    rawTermFreq.at(indices.at(i)) += weights.at(i);
    rawFreq.at(indices.at(i)) += 1;
    uniqueIndices.insert(indices.at(i));
  }

  // Compute tf
  float totalFreq = 0;
  for (size_t index : uniqueIndices) {
    rawTermFreq.at(index) = sqrt(rawTermFreq.at(index) / rawFreq.at(index));
    totalFreq += rawTermFreq.at(index);
  }
  for (size_t index : uniqueIndices) {
    rawTermFreq.at(index) = rawTermFreq.at(index) / totalFreq;
  }
  termFreq = af::array(rawTermFreq.size(), rawTermFreq.data());
}

void ir::IrInstance::computeScore(const af::array &bow, std::vector<float> &scores) {
  for (size_t batchId = 0; batchId < database_.size(); ++batchId) {
    af::array afScores = af::matmul(database_.at(batchId), bow);
    float* ptrScores = afScores.host<float>();
    scores.insert(scores.end(), ptrScores, ptrScores + afScores.dims(0));
  }
}

std::vector<ir::IrResult> ir::IrInstance::retrieve(const cv::Mat &image, int topK) {
  createInstanceIfNecessary();

  // Extract features from the image using perdoch's hessaff
  DLOG(INFO) << "Started extracting features from the query";
  boost::multi_array<float, 2> keypoints, descriptors;
  hesaff::extract(image.clone(), keypoints, descriptors, true);
  DLOG(INFO) << "Finished extracting features from the query";

  // Carry out quantization, using soft-assignment with `quantParams_`
  DLOG(INFO) << "Started quantizing the features";
  std::vector<size_t> indices;
  std::vector<float> weights;
  instance_->quantize(descriptors, indices, weights);
  DLOG(INFO) << "Finished quantizing the features";

  // Build bag-of-words vector
  DLOG(INFO) << "Started computing TF vector";
  af::array bow;
  instance_->computeTF(indices, weights, bow);
  DLOG(INFO) << "Finished computing TF vector";

  // Compute tfidf
  DLOG(INFO) << "Started computing TFIDF vector";
  bow *= instance_->invDocFreq_;
  bow *= instance_->invDocFreq_;
  DLOG(INFO) << "Finished computing TFIDF vector";

  // Compute scores
  DLOG(INFO) << "Started computing scores";
  std::vector<float> scores;
  instance_->computeScore(bow, scores);
  DLOG(INFO) << "Finished computing scores";

  // Sort scores and return
  std::vector<size_t> docIndices(scores.size());
  std::iota(docIndices.begin(), docIndices.end(), 0);

  std::sort(
    docIndices.begin(),
    docIndices.end(),
  [&scores](const size_t &lhs, const size_t &rhs) -> bool {
    return scores.at(lhs) > scores.at(rhs);
  });

  std::vector<ir::IrResult> result;
  if (topK == -1) {
    for (size_t id : docIndices) {
      result.push_back(ir::IrResult(instance_->docNames_.at(id), scores.at(id)));
    }
  } else {
    for (size_t i = 0; i < topK; ++i) {
      size_t id = docIndices.at(i);
      result.push_back(ir::IrResult(instance_->docNames_.at(id), scores.at(id)));
    }
  }
  return result;
}
