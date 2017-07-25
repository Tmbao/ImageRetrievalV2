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
 protected:

  // Static variables
  static std::shared_ptr<IrInstance> instance_;
  static boost::mutex initMutex_;
  static GlobalParams globalParams_;
  static QuantizationParams quantParams_;
  static DatabaseParams dbParams_;

  // Documents
  std::vector<std::string> docNames_;

  // Quantization variables
  flann::Index< flann::L2<double> >* quantIndex_;

  // Bag-of-word variables
  std::vector<af::array> database_;
  boost::mutex databaseMutex_;
  af::array sqrInvDocFreq_;

  // Construction methods

  template<typename IrType = IrInstance>
  static void createInstanceIfNecessary();

  void buildIndexIfNecessary();

  virtual void buildDatabase();

  void buildDatabaseOfBatchIfNecessary(
    const size_t &batchId,
    const size_t &fromDocId,
    const size_t &untilDocId,
    std::vector<double> &rawInvDocFreq,
    std::vector<boost::mutex> &rawInvMutex);

  void extractDbIfNecessary(
    const std::string &docName,
    boost::multi_array<double, 2> &keypoints,
    boost::multi_array<double, 2> &descriptors);

  void quantizeDbIfNecessary(
    const std::string &docName,
    std::vector<size_t> &indices,
    std::vector<double> &weights);

  void computeTFAndIndicesDbIfNecessary(
    const std::string &docName,
    std::vector<size_t> &indices,
    af::array &termFreq);

  //TODO: Update the flow to eliminate redundant tasks
  static void loadDocumentTask(
    IrInstance* instance,
    const size_t &batchId,
    const size_t &docId,
    std::vector<double> *rawInvDocFreq,
    std::vector<boost::mutex> *rawInvMutex);

  // Query methods

  static void loadQueryTask(
    const size_t &queryId,
    const cv::Mat* &image,
    af::array* &bows,
    boost::mutex* bowMutex);

  virtual void quantize(
    boost::multi_array<double, 2> &descriptors,
    std::vector<size_t> &termIndices,
    std::vector<double> &termWeights);

  virtual void computeTF(
    const std::vector<size_t> &indices,
    const std::vector<double> &weights,
    af::array &termFreq);

  virtual void computeScore(
    const af::array &bow,
    std::vector<double> &scores);

  virtual void computeScore(
    const af::array &bows,
    std::vector< std::vector<double> > &scores);

  virtual std::vector<IrResult> retrieveImpl(
    const cv::Mat& image,
    int topK);

  virtual std::vector< std::vector<IrResult> > retrieveImpl(
    const std::vector<cv::Mat> &images,
    int topK);

 public:
  IrInstance();

  /**
   * Creates an instance.
   */
  template<typename IrType = IrInstance>
  static void createInstanceIfNecessary(
    GlobalParams globalParams,
    QuantizationParams quantParams,
    DatabaseParams dbParams,
    bool reuse = true);

  /**
   * Retrieves a list of simiar images in the database sorted according to
   * their score.
   */
  template<typename IrType = IrInstance>
  static std::vector<IrResult> retrieve(const cv::Mat& image, int topK = -1);

  /**
   * Retrieves lists of similar images in the database sorted according to
   * their score.
   */
  template<typename IrType = IrInstance>
  static std::vector< std::vector<IrResult> > retrieve(
    const std::vector<cv::Mat> &images,
    int topK = -1);
};

}

template<typename IrType>
void ir::IrInstance::createInstanceIfNecessary() {
  boost::mutex::scoped_lock scopedLock(initMutex_);
  if (instance_ == nullptr) {
    instance_ = std::shared_ptr<IrType>(new IrType);

    instance_->buildIndexIfNecessary();
    instance_->buildDatabase();
  }
}

template<typename IrType>
void ir::IrInstance::createInstanceIfNecessary(
  GlobalParams globalParams,
  QuantizationParams quantParams,
  DatabaseParams dbParams,
  bool reuse) {
  boost::mutex::scoped_lock scopedLock(initMutex_);
  if (instance_ == nullptr || !reuse) {
    globalParams_ = globalParams;
    quantParams_ = quantParams;
    dbParams_ = dbParams;
    instance_ = std::shared_ptr<IrType>(new IrType);

    instance_->buildIndexIfNecessary();
    instance_->buildDatabase();
  }
}

template<typename IrType>
std::vector<ir::IrResult> ir::IrInstance::retrieve(const cv::Mat &image, int topK) {
  createInstanceIfNecessary<IrType>();
  return instance_->retrieveImpl(image, topK);
}

template<typename IrType>
std::vector< std::vector<ir::IrResult> > ir::IrInstance::retrieve(
  const std::vector<cv::Mat> &images,
  int topK) {
  createInstanceIfNecessary<IrType>();
  return instance_->retrieveImpl(images, topK);
}

#endif /* ir_instance_h */
