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
  DatabaseParams dbParams) {
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
  LOG(INFO) << "Started reading codebook file";
  flann::Matrix<double> codebook;
  flann::load_from_file(
    codebook,
    quantParams_.codebookFile,
    quantParams_.codebookName);
  LOG(INFO) << "Finished reading codebook file";

  LOG(INFO) << "Started creating index";
  flann::IndexParams* indexParams;
  if (!globalParams_.overwrite && boost::filesystem::exists(quantParams_.indexFile)) {
    indexParams = new flann::SavedIndexParams(quantParams_.indexFile);
  } else {
    indexParams = new flann::KDTreeIndexParams(quantParams_.nTrees);
  }
  LOG(INFO) << "Finished creating index";

  LOG(INFO) << "Started constructing KDTree";
  quantIndex_ = new flann::Index< flann::L2<double> >(codebook, *indexParams);
  quantIndex_->buildIndex();
  quantIndex_->save(quantParams_.indexFile);
  LOG(INFO) << "Finished constructing KDTree";
}

void ir::IrInstance::extractDbIfNecessary(
  const std::string &docName,
  boost::multi_array<double, 2> &keypoints,
  boost::multi_array<double, 2> &descriptors) {

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
  std::vector<size_t> &indices,
  std::vector<double> &weights) {

  std::string indexPath = dbParams_.getFullPath(docName, INDEX);
  std::string weightPath = dbParams_.getFullPath(docName, WEIGHT);

  // Check if quantization data exists
  if (boost::filesystem::exists(indexPath) &&
      boost::filesystem::exists(weightPath) &&
      !globalParams_.overwrite) {
    load(indices, indexPath);
    load(weights, weightPath);
  } else {
    // Quantize the document
    boost::multi_array<double, 2> keypoints, descriptors;
    extractDbIfNecessary(docName, keypoints, descriptors);

    if (keypoints.size() > 0) {
      quantize(descriptors, indices, weights);
    }
    save(indices, indexPath);
    save(weights, weightPath);
  }
}

void ir::IrInstance::computeTFAndIndicesDbIfNecessary(
  const std::string &docName,
  std::vector<size_t> &indices,
  af::array &termFreq) {

  std::string indexPath = dbParams_.getFullPath(docName, INDEX);
  std::string tfPath = dbParams_.getFullPath(docName, TERM_FREQUENCY);

  // Check if tf and index exist
  if (boost::filesystem::exists(indexPath) &&
      boost::filesystem::exists(tfPath) &&
      !globalParams_.overwrite && false) {
    load(indices, indexPath);
    termFreq = af::readArray(tfPath.c_str(), "");
  } else {
    // Quantize the document
    std::vector<double> weights;
    quantizeDbIfNecessary(docName, indices, weights);

    if (indices.size() > 0) {
      computeTF(indices, weights, termFreq);
    } else {
      termFreq = af::constant(0, dbParams_.nWords, f64);
    }
    af::saveArray("", termFreq, tfPath.c_str());
  }
}

void ir::IrInstance::loadDocumentTask(
  IrInstance* instance,
  const size_t &batchId,
  const size_t &docId,
  std::vector<double> *rawInvDocFreq,
  std::vector<boost::mutex> *rawInvMutex) {

  std::string docName = instance->docNames_.at(docId);
  LOG(INFO) << "Started loading document #" + std::to_string(docId) + " " + docName;

  // Make sure that all threads use a single device
  af::setDevice(0);

  // Build bag-of-words vector
  std::vector<size_t> indices;
  af::array termFreq;
  instance->computeTFAndIndicesDbIfNecessary(docName, indices, termFreq);

  // Update database
  instance->databaseMutex_.lock();
  instance->database_.at(batchId).row(docId - batchId * dbParams_.batchSize) = termFreq;
  instance->databaseMutex_.unlock();

  // Add indices to inverted index and update idf
  for (size_t i = 0; i < indices.size(); ++i) {
    rawInvMutex->at(indices.at(i)).lock();
    rawInvDocFreq->at(indices.at(i)) += 1;
    rawInvMutex->at(indices.at(i)).unlock();
  }

  LOG(INFO) << "Finished loading document #" + std::to_string(docId) + " " + docName;
}

void ir::IrInstance::buildDatabaseOfBatchIfNecessary(
  const size_t &batchId,
  const size_t &fromDocId,
  const size_t &untilDocId,
  std::vector<double> &rawInvDocFreq,
  std::vector<boost::mutex> &rawInvMutex) {

  /**
   The following stuffs will be done sequentially and in parallel
   - Extract features (hesaff)
   - Quantize (quantize)
   - Compute tf (computeTF)
   - Compute idf (invDocFreq_)
   */

  LOG(INFO) << "Started building database #" << batchId;
  database_.at(batchId) = af::constant(0, untilDocId - fromDocId, dbParams_.nWords, f64);

  // Initialize a thread pool
  boost::thread_group threads;
  boost::asio::io_service ioService;

  LOG(INFO) << "Started adding tasks, nTasks = " << untilDocId - fromDocId;
  for (size_t docId = fromDocId; docId < untilDocId; ++docId) {
    ioService.post(boost::bind(
                     loadDocumentTask,
                     this,
                     batchId,
                     docId,
                     &rawInvDocFreq,
                     &rawInvMutex));
  }
  LOG(INFO) << "Finished adding tasks, nTasks = " << untilDocId - fromDocId;

  LOG(INFO) << "Started adding threads, nThreads = " << globalParams_.nThreads;
  for (size_t i = 0; i < globalParams_.nThreads; ++i) {
    threads.create_thread(boost::bind(&boost::asio::io_service::run, &ioService));
  }
  LOG(INFO) << "Finished adding threads, nThreads = " << globalParams_.nThreads;

  threads.join_all();
  ioService.stop();
  LOG(INFO) << "Finished all tasks";

  LOG(INFO) << "Started compressing database #" << batchId;
  database_.at(batchId) = af::sparse(database_.at(batchId));
  LOG(INFO) << "Finished building database #" << batchId;
}

void ir::IrInstance::buildDatabase() {
  docNames_ = dbParams_.getDocuments();
  size_t nDocs = docNames_.size();

  LOG(INFO) << "Number of documents = " << nDocs;

  database_.resize((nDocs - 1) / dbParams_.batchSize + 1);
  std::vector<double> rawInvDocFreq(dbParams_.nWords);
  std::vector<boost::mutex> rawInvMutex(dbParams_.nWords);

  for (size_t batchId = 0, fromDocId = 0;
       fromDocId < nDocs;
       fromDocId += dbParams_.batchSize, ++batchId) {
    size_t untilDocId = std::min(fromDocId + (size_t) dbParams_.batchSize, nDocs);
    buildDatabaseOfBatchIfNecessary(
      batchId,
      fromDocId,
      untilDocId,
      rawInvDocFreq,
      rawInvMutex);
  }

  // Compute idf
  LOG(INFO) << "Started computing idf";
  for (size_t i = 0; i < rawInvDocFreq.size(); ++i) {
    if (rawInvDocFreq.at(i) > 0) {
      rawInvDocFreq.at(i) = log(nDocs / rawInvDocFreq.at(i));
    }
  }
  invDocFreq_ = af::array(rawInvDocFreq.size(), rawInvDocFreq.data());
  invDocFreq_ = invDocFreq_ * invDocFreq_;
  LOG(INFO) << "Finished computing idf";
}

void ir::IrInstance::quantize(
  boost::multi_array<double, 2> &descriptors,
  std::vector<size_t> &termIndices,
  std::vector<double> &termWeights) {

  flann::Matrix<double> queries(
    descriptors.data(),
    descriptors.shape()[0],
    descriptors.shape()[1]);

  flann::Matrix<int> indices(
    new int[descriptors.shape()[0] * quantParams_.knn],
    descriptors.shape()[0],
    quantParams_.knn);

  flann::Matrix<double> dists(
    new double[descriptors.shape()[0] * quantParams_.knn],
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

  // Fetch weights, apply radial weighting and normalize
  int* ptrIndices = indices.ptr();
  double* ptrWeights = dists.ptr();

  termIndices = std::vector<size_t>(ptrIndices, ptrIndices + indices.rows * indices.cols);
  termWeights = std::vector<double>(ptrWeights, ptrWeights + dists.rows * dists.cols);

  for (size_t i = 0; i < dists.rows * dists.cols; i += dists.cols) {
    double sum = 0;
    for (size_t j = 0; j < i + dists.cols; ++j) {
      termWeights.at(j) = exp(-termWeights.at(j) / (2 * quantParams_.deltaSqr));
      sum += termWeights.at(j);
    }
    for (size_t j = 0; j < i + dists.cols; ++j) {
      termWeights.at(j) /= sum;
    }
  }
}

void ir::IrInstance::computeTF(
  const std::vector<size_t> &indices,
  const std::vector<double> &weights,
  af::array &termFreq) {

  // Join indices and weights
  std::vector<double> rawTermFreq(dbParams_.nWords, 0);
  std::vector<size_t> rawFreq(dbParams_.nWords, 0);
  std::set<size_t> uniqueIndices;
  for (size_t i = 0; i < indices.size(); ++i) {
    rawTermFreq.at(indices.at(i)) += weights.at(i);
    rawFreq.at(indices.at(i)) += 1;
    uniqueIndices.insert(indices.at(i));
  }

  // Compute tf
  double totalFreq = 0;
  for (size_t index : uniqueIndices) {
    rawTermFreq.at(index) /= sqrt(rawFreq.at(index));
    totalFreq += rawTermFreq.at(index);
  }
  for (size_t index : uniqueIndices) {
    rawTermFreq.at(index) = sqrt(rawTermFreq.at(index) / totalFreq);
  }
  termFreq = af::array(rawTermFreq.size(), rawTermFreq.data());
}

void ir::IrInstance::computeScore(const af::array &bow, std::vector<double> &scores) {
  for (size_t batchId = 0; batchId < database_.size(); ++batchId) {
    af::array afScores = af::matmul(database_.at(batchId), bow);
    double* ptrScores = afScores.host<double>();
    scores.insert(scores.end(), ptrScores, ptrScores + afScores.dims(0));
    af::freeHost(ptrScores);
  }
}

std::vector<ir::IrResult> ir::IrInstance::retrieve(const cv::Mat &image, int topK) {
  createInstanceIfNecessary();

  // Extract features from the image using perdoch's hessaff
  boost::multi_array<double, 2> keypoints, descriptors;
  hesaff::extract(image.clone(), keypoints, descriptors, true);

  // Carry out quantization, using soft-assignment with `quantParams_`
  std::vector<size_t> indices;
  std::vector<double> weights;
  instance_->quantize(descriptors, indices, weights);

  // Compute tf
  af::array bow;
  instance_->computeTF(indices, weights, bow);

  // Compute tfidf
  bow *= instance_->invDocFreq_;

  // Compute scores
  std::vector<double> scores;
  instance_->computeScore(bow, scores);

  // Sort scores and return
  std::vector<ir::IrResult> result;

  std::vector<size_t> docIndices(scores.size());
  std::iota(docIndices.begin(), docIndices.end(), 0);

  std::sort(
    docIndices.begin(),
    docIndices.end(),
  [&scores](const size_t &lhs, const size_t &rhs) -> bool {
    return scores.at(lhs) > scores.at(rhs);
  });

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

void ir::IrInstance::computeScore(
  const af::array &bows,
  std::vector< std::vector<double> > &scores) {

  for (size_t batchId = 0; batchId < database_.size(); ++batchId) {
    af::array afScores = af::matmul(database_.at(batchId), bows);
    double* ptrScores = afScores.host<double>();

    size_t batchSize = afScores.dims(0);
    for (size_t i = 0; i < scores.size(); ++i) {
      scores.at(i).insert(
        scores.at(i).end(),
        ptrScores + i * batchSize,
        ptrScores + (i + 1) * batchSize);
    }

    af::freeHost(ptrScores);
  }
}

void ir::IrInstance::loadQueryTask(
  const size_t &queryId,
  const cv::Mat* &image,
  af::array* &bows,
  boost::mutex* bowMutex) {
  // Extract features from the image using perdoch's hessaff
  boost::multi_array<double, 2> keypoints, descriptors;
  hesaff::extract(image->clone(), keypoints, descriptors, true);

  // Carry out quantization, using soft-assignment with `quantParams_`
  std::vector<size_t> indices;
  std::vector<double> weights;
  instance_->quantize(descriptors, indices, weights);

  // Compute tf
  af::array tf;
  instance_->computeTF(indices, weights, tf);

  // Compute tfidf
  tf *= instance_->invDocFreq_;

  bowMutex->lock();
  bows->col(queryId) = tf;
  bowMutex->unlock();
}

std::vector< std::vector<ir::IrResult> > ir::IrInstance::retrieve(
  const std::vector<cv::Mat> &images,
  int topK) {

  assert(images.size() > 0);
  assert(images.size() <= dbParams_.batchSize);

  af::array bows(dbParams_.nWords, images.size(), f64);
  boost::mutex bowMutex;

  // Initialize a thread pool
  boost::thread_group threads;
  boost::asio::io_service ioService;

  // Initialize tasks
  for (size_t queryId = 0; queryId < images.size(); ++queryId) {
    ioService.post(boost::bind(
                     loadQueryTask,
                     queryId,
                     &images.at(queryId),
                     &bows,
                     &bowMutex));
  }

  // Initialize threads
  for (size_t i = 0; i < globalParams_.nThreads; ++i) {
    threads.create_thread(boost::bind(&boost::asio::io_service::run, &ioService));
  }

  // Join tasks
  threads.join_all();
  ioService.stop();

  // Compute scores
  std::vector< std::vector<double> > scores(images.size());
  instance_->computeScore(bows, scores);

  // Sort scores and return
  std::vector< std::vector<ir::IrResult> > results(scores.size());

  std::vector<size_t> docIndices(scores.at(0).size());
  std::iota(docIndices.begin(), docIndices.end(), 0);
  for (size_t i = 0; i < scores.size(); ++i) {
    std::sort(
      docIndices.begin(),
      docIndices.end(),
    [&scores, i](const size_t &lhs, const size_t &rhs) -> bool {
      return scores.at(i).at(lhs) > scores.at(i).at(rhs);
    });

    if (topK == -1) {
      for (size_t id : docIndices) {
        results.at(i).push_back(ir::IrResult(instance_->docNames_.at(id), scores.at(i).at(id)));
      }
    } else {
      for (size_t j = 0; j < topK; ++j) {
        size_t id = docIndices.at(j);
        results.at(i).push_back(ir::IrResult(instance_->docNames_.at(id), scores.at(i).at(id)));
      }
    }
  }

  return results;
}
