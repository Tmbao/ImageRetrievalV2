//
//  task_status.h
//  server
//
//  Created by Bao Truong on 7/12/17.
//
//

#ifndef task_status_h
#define task_status_h

namespace taskmgr {
enum TaskStatus {
  IN_QUEUE,
  PROCESSING,
  READY,
  UNKNOWN
};
}

#endif /* task_status_h */
