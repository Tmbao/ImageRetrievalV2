//
//  task_manager.cpp
//  server
//
//  Created by Bao Truong on 7/12/17.
//
//

#include "task_manager.h"

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
