//
//  ir_instance.h
//  server
//
//  Created by Bao Truong on 7/3/17.
//
//

#ifndef ir_instance_h
#define ir_instance_h

#include <arrayfire.h>
#include <boost/container/vector.hpp>
#include <boost/container/set.hpp>
#include <boost/thread.hpp>
#include <flann/flann.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "../global_params.h"
#include "database_params.h"
#include "ir_result.h"
#include "quantization_params.h"


namespace ir {

typedef int size_t;

/**
 * A thread safe image retrieval system.
 */
class IrInstance {
 private:

  // Static variables
  static IrInstance* instance_;
  static boost::mutex initMutex_;
  static GlobalParams globalParams_;
  static QuantizationParams quantParams_;
  static DatabaseParams dbParams_;

  // Documents
  boost::container::vector<std::string> docNames_;

  // Quantization variables
  flann::Index< flann::L2<float> >* quantIndex_;

  // Bag-of-word variables
  af::array database_;
  boost::container::vector<af::array> invIndex_;
  af::array invDocFreq_;

  IrInstance();

  // Construction methods
  
  void createInstanceIfNecessary();

  void buildIndexIfNecessary();

  void buildDatabase();

  void extractDbIfNecessary(
    const std::string &docName,
    af::array &keypoints,
    af::array &descriptors);

  void quantizeDbIfNecessary(
    const std::string &docName,
    const af::array &descriptors,
    std::vector<size_t> &indices,
    std::vector<float> &weights);

  void computeTFDbIfNecessary(
    const std::string &docName,
    const std::vector<size_t> &indices,
    const std::vector<float> &weights,
    af::array &termFreq);

  static void loadDocumentTask(
    IrInstance* &instance,
    boost::container::vector< boost::container::set<size_t> > &rawInvIndex,
    const size_t &docId);

  // Query methods

  void quantize(
    const af::array &descriptors,
    std::vector<size_t> &termIndices,
    std::vector<float> &termWeights);

  void computeTF(
    const std::vector<size_t> &indices,
    const std::vector<float> &weights,
    af::array &termFreq);

  void computeScore(
    const af::array& bow,
    boost::container::vector<float> &scores);

 public:

  /**
   * Creates an instance.
   */
  void createInstanceIfNecessary(
    GlobalParams globalParams,
    QuantizationParams quantParams,
    DatabaseParams dbParams
  );

  /**
   * Retrieves a list of simiar image in the database along with their score.
   */
  boost::container::vector<IrResult> retrieve(const cv::Mat& image, int topK = -1);
};

}

#endif /* ir_instance_h */
