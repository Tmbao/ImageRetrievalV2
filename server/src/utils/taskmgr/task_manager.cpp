//
//  task_manager.cpp
//  server
//
//  Created by Bao Truong on 7/12/17.
//
//

#include "task_manager.h"

std::shared_ptr<taskmgr::TaskManager> taskmgr::TaskManager::instance_ = nullptr;
GlobalParams taskmgr::TaskManager::globalParams_;

void taskmgr::TaskManager::createInstanceIfNecessary() {
  boost::mutex::scoped_lock scopedLock(instance_->initMutex_);
  if (instance_ == nullptr) {
    instance_ = std::shared_ptr<taskmgr::TaskManager>(new taskmgr::TaskManager());
  }
}

bool taskmgr::TaskManager::addTask(const std::string &id, const cv::Mat &mat) {
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

void taskmgr::TaskManager::execute() {
  createInstanceIfNecessary();
  boost::mutex::scoped_lock scopedLock(instance_->addTaskMutex_);
  executeAsync();
}

void taskmgr::TaskManager::executeAsync() {
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

void taskmgr::TaskManager::executeSync(
  const std::vector<std::string> &identities,
  const std::vector<cv::Mat> &matrices) {
  instance_->executionMutex_.lock();
  std::vector< std::vector<ir::IrResult> > ranklists = ir::IrInstance::retrieve(matrices);
  instance_->executionMutex_.unlock();

  instance_->updateResultMutex_.lock();
  for (size_t i = 0; i < ranklists.size(); ++i) {
    instance_->statuses_.at(identities.at(i)) = READY;
    instance_->results_.at(identities.at(i)) = ranklists.at(i);
  }
  instance_->updateResultMutex_.unlock();
}

taskmgr::TaskStatus taskmgr::TaskManager::fetchResult(
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
