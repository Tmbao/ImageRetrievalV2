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


af::array ir::IrAverageQueryExpansion::enrichQuery(const std::vector<size_t> &indices) {
  std::vector<size_t> chosenIndices(
    indices.begin(),
    min(indices.end(), indices.begin() + qeParams_.nChosen));
  sort(chosenIndices.begin(), chosenIndices.end());

  af::array newQuery = af::constant(0, dbParams_.nWords, f64);
  for (size_t batchId = 0, offset = 0, chosenId = 0, evalStep = 0;
       batchId < database_.size();
       ++batchId, offset += globalParams_.batchSize) {
    size_t batchSize = database_.at(batchId).dims(0);

    std::vector<double> dataIndices(batchSize, 0);
    bool isEmpty = true;
    while (chosenId < chosenIndices.size() &&
           chosenIndices.at(chosenId) >= offset &&
           chosenIndices.at(chosenId) < offset + batchSize) {
      dataIndices.at(chosenIndices.at(chosenId) - offset) = (double) 1 / chosenIndices.size();
      ++chosenId;
      isEmpty = false;
    }
    if (isEmpty) {
      ++evalStep;
      newQuery = newQuery + af::matmulTN(
                   database_.at(batchId),
                   af::array(dataIndices.size(), dataIndices.data()));

      if (evalStep % qeParams_.evalPeriod == 0) {
        af::eval(newQuery);
      }
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
  
  assert(scores.size() == docIndices.size());

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
