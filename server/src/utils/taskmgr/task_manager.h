//
//  task_manager.h
//  server
//
//  Created by Bao Truong on 7/12/17.
//
//

#ifndef task_manager_h
#define task_manager_h

#include <boost/thread.hpp>
#include <memory>

#include "../global_params.h"
#include "../ir/ir_instance.h"
#include "task_status.h"

namespace taskmgr {

/**
 * A thread-safe IR task management class.
 */
template<typename IrType = ir::IrInstance>
class TaskManager {
 private:
  static std::shared_ptr< TaskManager<IrType> > instance_;
  static GlobalParams globalParams_;

  boost::mutex initMutex_;
  boost::mutex executionMutex_;
  boost::mutex addTaskMutex_;
  boost::mutex updateResultMutex_;

  std::unordered_map< std::string, std::vector<ir::IrResult> > results_;
  std::unordered_map<std::string, TaskStatus> statuses_;

  std::vector<std::string> identities_;
  std::vector<cv::Mat> matrices_;

  TaskManager() {}

  static void createInstanceIfNecessary();

  static void executeSync(
    const std::vector<std::string> &identities,
    const std::vector<cv::Mat> &matrices);

  static void executeAsync();

 public:
  /**
   * Adds a retrieval task to queue. For performance purpose, this function
   * will not execute the retrieval process unless the queue size matches global
   * batch size.
   * @id  The id of the task, this is also the place where the result of
   * this task will be stored.
   * @mat The image.
   * This function returns false when the id has already existed.
   */
  static bool addTask(const std::string &id, const cv::Mat &mat);

  /**
   * Executes the retrieval process on the queue. This function is non-blocking.
   */
  static void execute();

  /**
   * Fetchs the result of a query if it is ready.
   * Returns the status of the query.
   */
  static TaskStatus fetchResult(
    const std::string &id,
    std::vector<ir::IrResult> &result);
};

}

template<typename IrType>
std::shared_ptr< taskmgr::TaskManager<IrType> > taskmgr::TaskManager<IrType>::instance_ = nullptr;

template<typename IrType>
GlobalParams taskmgr::TaskManager<IrType>::globalParams_;

template<typename IrType>
void taskmgr::TaskManager<IrType>::createInstanceIfNecessary() {
  boost::mutex::scoped_lock scopedLock(instance_->initMutex_);
  if (instance_ == nullptr) {
    instance_ = std::shared_ptr< taskmgr::TaskManager<IrType> >(
                                                                new taskmgr::TaskManager<IrType>());
  }
}

template<typename IrType>
bool taskmgr::TaskManager<IrType>::addTask(const std::string &id, const cv::Mat &mat) {
  createInstanceIfNecessary();
  
  boost::mutex::scoped_lock scopedLock(instance_->addTaskMutex_);
  
  if (instance_->statuses_.count(id)) {
    return false;
  }
  
  instance_->statuses_.at(id) = IN_QUEUE;
  instance_->identities_.push_back(id);
  instance_->matrices_.push_back(mat);
  
  if (instance_->identities_.size() == globalParams_.batchSize) {
    executeAsync();
  }
  return true;
}

template<typename IrType>
void taskmgr::TaskManager<IrType>::execute() {
  createInstanceIfNecessary();
  boost::mutex::scoped_lock scopedLock(instance_->addTaskMutex_);
  executeAsync();
}

template<typename IrType>
void taskmgr::TaskManager<IrType>::executeAsync() {
  std::vector<std::string> localIdentities(instance_->identities_);
  std::vector<cv::Mat> localMatrices(instance_->matrices_);
  
  instance_->updateResultMutex_.lock();
  for (std::string identity : localIdentities) {
    instance_->statuses_.at(identity) = PROCESSING;
  }
  instance_->updateResultMutex_.unlock();
  boost::thread executionThread(executeSync, localIdentities, localMatrices);
  
  instance_->identities_.clear();
  instance_->matrices_.clear();
}

template<typename IrType>
void taskmgr::TaskManager<IrType>::executeSync(
                                               const std::vector<std::string> &identities,
                                               const std::vector<cv::Mat> &matrices) {
  instance_->executionMutex_.lock();
  std::vector< std::vector<ir::IrResult> > ranklists = ir::IrInstance::retrieve<IrType>(matrices);
  instance_->executionMutex_.unlock();
  
  instance_->updateResultMutex_.lock();
  for (size_t i = 0; i < ranklists.size(); ++i) {
    instance_->statuses_.at(identities.at(i)) = READY;
    instance_->results_.at(identities.at(i)) = ranklists.at(i);
  }
  instance_->updateResultMutex_.unlock();
}

template<typename IrType>
taskmgr::TaskStatus taskmgr::TaskManager<IrType>::fetchResult(
                                                              const std::string &id,
                                                              std::vector<ir::IrResult> &result) {
  createInstanceIfNecessary();
  
  boost::mutex::scoped_lock scopedLockAddTask(instance_->addTaskMutex_);
  boost::mutex::scoped_lock scopedLockResult(instance_->updateResultMutex_);
  
  if (instance_->statuses_.count(id) == 0) {
    return UNKNOWN;
  }
  
  taskmgr::TaskStatus status = instance_->statuses_.at(id);
  if (status == READY) {
    result = instance_->results_.at(id);
  }
  return status;
}

#endif /* task_manager_h */
