//
//  ir_instance.h
//  server
//
//  Created by Bao Truong on 7/3/17.
//
//

#ifndef ir_instance_h
#define ir_instance_h

#define FLANN_USE_OPENCL

#include <arrayfire.h>
#include <boost/multi_array.hpp>
#include <boost/thread.hpp>
#include <flann/flann.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <set>
#include <vector>

#include "../global_params.h"
#include "database_params.h"
#include "ir_result.h"
#include "quantization_params.h"


namespace ir {

// Arrayfire does not accept unsigned long
typedef int size_t;

/**
 * A thread safe image retrieval system.
 */
class IrInstance {
 private:

  // Static variables
  static std::shared_ptr<IrInstance> instance_;
  static boost::mutex initMutex_;
  static GlobalParams globalParams_;
  static QuantizationParams quantParams_;
  static DatabaseParams dbParams_;

  // Documents
  std::vector<std::string> docNames_;

  // Quantization variables
  flann::Index< flann::L2<float> >* quantIndex_;

  // Bag-of-word variables
  std::vector<af::array> database_;
  boost::mutex databaseMutex_;
  std::vector<af::array> invIndex_;
  af::array invDocFreq_;

  IrInstance();

  // Construction methods

  static void createInstanceIfNecessary();

  void buildIndexIfNecessary();

  void buildDatabase();

  void buildDatabaseOfBatchIfNecessary(
    const size_t &batchId,
    const size_t &fromDocId,
    const size_t &untilDocId,
    std::vector< std::set<size_t> > &rawInvIndex,
    std::vector<float> &rawInvDocFreq,
    std::vector<boost::mutex> &rawInvMutex
  );

  void extractDbIfNecessary(
    const std::string &docName,
    boost::multi_array<float, 2> &keypoints,
    boost::multi_array<float, 2> &descriptors);

  void quantizeDbIfNecessary(
    const std::string &docName,
    boost::multi_array<float, 2> &descriptors,
    std::vector<size_t> &indices,
    std::vector<float> &weights);

  void computeTFDbIfNecessary(
    const std::string &docName,
    const std::vector<size_t> &indices,
    const std::vector<float> &weights,
    af::array &termFreq);

  //TODO: Update the flow to eliminate redundant tasks
  static void loadDocumentTask(
    IrInstance* &instance,
    const size_t &batchId,
    const size_t &docId,
    std::vector<float> *rawInvDocFreq,
    std::vector<boost::mutex> *rawInvMutex,
    std::vector< std::set<size_t> > *rawInvIndex);

  // Query methods

  void quantize(
    boost::multi_array<float, 2> &descriptors,
    std::vector<size_t> &termIndices,
    std::vector<float> &termWeights);

  void computeTF(
    const std::vector<size_t> &indices,
    const std::vector<float> &weights,
    af::array &termFreq);

  void computeScore(
    const af::array& bow,
    std::vector<float> &scores);

 public:

  /**
   * Creates an instance.
   */
  static void createInstanceIfNecessary(
    GlobalParams globalParams,
    QuantizationParams quantParams,
    DatabaseParams dbParams
  );

  /**
   * Retrieves a list of simiar image in the database sorted according to
   * their score.
   */
  static std::vector<IrResult> retrieve(const cv::Mat& image, int topK = -1);
};

}

#endif /* ir_instance_h */
