//
//  ir_aqe_extension.cpp
//  server
//
//  Created by Bao Truong on 7/24/17.
//
//

#include "ir_aqe_extension.h"

#include <glog/logging.h>

#include "../../hesaff/hesaff.h"

void ir::IrAverageQueryExpansion::buildDatabase() {
  docNames_ = dbParams_.getDocuments();
  size_t nDocs = docNames_.size();

  LOG(INFO) << "Number of documents = " << nDocs;

  database_.resize((nDocs - 1) / globalParams_.batchSize + 1);
  databaseTranspose_.resize(database_.size());
  std::vector<double> rawInvDocFreq(dbParams_.nWords);
  std::vector<boost::mutex> rawInvMutex(dbParams_.nWords);

  for (size_t batchId = 0, fromDocId = 0;
       fromDocId < nDocs;
       fromDocId += globalParams_.batchSize, ++batchId) {
    size_t untilDocId = std::min(fromDocId + (size_t) globalParams_.batchSize, nDocs);
    buildDatabaseOfBatchIfNecessary(
      batchId,
      fromDocId,
      untilDocId,
      rawInvDocFreq,
      rawInvMutex);

    databaseTranspose_.at(batchId) = af::transpose(database_.at(batchId));
    LOG(INFO) << "Started compressing database #" << batchId;
    databaseTranspose_.at(batchId) = af::sparse(databaseTranspose_.at(batchId));
    database_.at(batchId) = af::sparse(database_.at(batchId));
    LOG(INFO) << "Finished building database #" << batchId;
  }

  // Compute idf
  LOG(INFO) << "Started computing idf";
  for (size_t i = 0; i < rawInvDocFreq.size(); ++i) {
    if (rawInvDocFreq.at(i) > 0) {
      rawInvDocFreq.at(i) = log(nDocs / rawInvDocFreq.at(i));
    }
  }
  sqrInvDocFreq_ = af::array(rawInvDocFreq.size(), rawInvDocFreq.data());
  sqrInvDocFreq_ = sqrInvDocFreq_ * sqrInvDocFreq_;
  LOG(INFO) << "Finished computing idf";
}

af::array ir::IrAverageQueryExpansion::enrichQuery(const std::vector<size_t> &indices) {
  std::vector<size_t> chosenIndices(
    indices.begin(),
    min(indices.end(), indices.begin() + qeParams_.nChosen));
  sort(chosenIndices.begin(), chosenIndices.end());

  af::array newQuery = af::constant(0, dbParams_.nWords, f64);
  for (size_t batchId = 0, offset = 0, chosenId = 0;
       batchId < database_.size();
       ++batchId, offset += globalParams_.batchSize) {
    size_t batchSize = database_.at(batchId).dims(0);
    af::array afIndices = af::constant(0, batchSize, f64);
    while (chosenId < chosenIndices.size() &&
           chosenIndices.at(chosenId) >= offset &&
           chosenIndices.at(chosenId) < offset + batchSize) {
      afIndices(chosenIndices.at(chosenId) - offset) = (double) 1 / chosenIndices.size();
      ++chosenId;
    }
    DLOG(INFO) << "Enriching #" << batchId;
    newQuery = newQuery + af::matmul(databaseTranspose_.at(batchId), afIndices);

    if (batchId % qeParams_.evalPeriod == 0) {
      af::eval(newQuery);
    }
  }
  return newQuery;
}

std::vector<ir::IrResult> ir::IrAverageQueryExpansion::retrieveImpl(
  const cv::Mat& image,
  int topK) {
  // Extract features from the image using perdoch's hessaff
  boost::multi_array<double, 2> keypoints, descriptors;
  hesaff::extract(image.clone(), keypoints, descriptors, true);

  // Carry out quantization, using soft-assignment with `quantParams_`
  std::vector<size_t> indices;
  std::vector<double> weights;
  quantize(descriptors, indices, weights);

  // Compute tf
  af::array bow;
  computeTF(indices, weights, bow);

  // Compute tfidf
  bow *= sqrInvDocFreq_;

  // Compute scores
  std::vector<double> scores;
  computeScore(bow, scores);

  // Sort scores
  std::vector<ir::IrResult> result;

  std::vector<size_t> docIndices(scores.size());
  std::iota(docIndices.begin(), docIndices.end(), 0);

  std::sort(
    docIndices.begin(),
    docIndices.end(),
  [&scores](const size_t &lhs, const size_t &rhs) -> bool {
    return scores.at(lhs) > scores.at(rhs);
  });

  // Carryout average query expansion
  bow = enrichQuery(docIndices);
  computeScore(bow, scores);

  std::sort(
    docIndices.begin(),
    docIndices.end(),
  [&scores](const size_t &lhs, const size_t &rhs) -> bool {
    return scores.at(lhs) > scores.at(rhs);
  });

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

std::vector< std::vector<ir::IrResult> > ir::IrAverageQueryExpansion::retrieveImpl(
  const std::vector<cv::Mat> &images,
  int topK) {
  return std::vector< std::vector<ir::IrResult> >();
}
