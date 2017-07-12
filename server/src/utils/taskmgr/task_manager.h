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
class TaskManager {
 private:
  static std::shared_ptr<TaskManager> instance_;
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
   * Execute the retrieval process on the queue. This function is non-blocking.
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

#endif /* task_manager_h */
