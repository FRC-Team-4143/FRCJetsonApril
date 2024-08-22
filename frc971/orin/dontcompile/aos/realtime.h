#ifndef AOS_REALTIME_H_
#define AOS_REALTIME_H_

#include <sched.h>

#include <ostream>
#include <string_view>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"

// Stringifies the cpu_set_t for LOG().
template <typename Sink>
void AbslStringify(Sink &sink, const cpu_set_t &cpuset) {
  sink.Append("{CPUs ");
  bool first_found = false;
  for (int i = 0; i < CPU_SETSIZE; ++i) {
    if (CPU_ISSET(i, &cpuset)) {
      if (first_found) {
        sink.Append(", ");
      }
      absl::Format(&sink, "%d", i);
      first_found = true;
    }
  }
  sink.Append("}");
}

namespace aos {

// Locks everything into memory and sets the limits.  This plus InitNRT are
// everything you need to do before SetCurrentThreadRealtimePriority will make
// your thread RT.  Called as part of ShmEventLoop::Run()
void InitRT();

// Sets up this process to write core dump files.
// This is called by Init*, but it's here for other files that want this
// behavior without calling Init*.
void WriteCoreDumps();

void LockAllMemory();

void ExpandStackSize();

// Sets the name of the current thread.
// This will displayed by `top -H`, dump_rtprio, and show up in logs.
// name can have a maximum of 16 characters.
void SetCurrentThreadName(const std::string_view name);

// Creates a cpu_set_t from a list of CPUs.
inline cpu_set_t MakeCpusetFromCpus(std::initializer_list<int> cpus) {
  cpu_set_t result;
  CPU_ZERO(&result);
  for (int cpu : cpus) {
    CPU_SET(cpu, &result);
  }
  return result;
}

// Returns the affinity representing all the CPUs.
inline cpu_set_t DefaultAffinity() {
  cpu_set_t result;
  for (int i = 0; i < CPU_SETSIZE; ++i) {
    CPU_SET(i, &result);
  }
  return result;
}

// Returns the current thread's CPU affinity.
cpu_set_t GetCurrentThreadAffinity();

// Sets the current thread's scheduling affinity.
void SetCurrentThreadAffinity(const cpu_set_t &cpuset);

// Everything below here needs AOS to be initialized before it will work
// properly.

// Sets the current thread's realtime priority.
void SetCurrentThreadRealtimePriority(int priority);

// Unsets all threads realtime priority in preparation for exploding.
void FatalUnsetRealtimePriority();

// Sets the current thread back down to non-realtime priority.
void UnsetCurrentThreadRealtimePriority();

// Registers our hooks which crash on RT malloc.
void RegisterMallocHook();

// CHECKs that we are (or are not) running on the RT scheduler.  Useful for
// enforcing that operations which are or are not bounded shouldn't be run. This
// works both in simulation and when running against the real target.
void CheckRealtime();
void CheckNotRealtime();

// Marks that we are or are not running on the realtime scheduler.  Returns the
// previous state.
//
// Note: this shouldn't be used directly.  The event loop primitives should be
// used instead.
bool MarkRealtime(bool realtime);

// Class which restores the current RT state when destructed.
class ScopedRealtimeRestorer {
 public:
  ScopedRealtimeRestorer();
  ~ScopedRealtimeRestorer() { MarkRealtime(prior_); }

 private:
  const bool prior_;
};

// Class which marks us as on the RT scheduler until it goes out of scope.
// Note: this shouldn't be needed for most applications.
class ScopedRealtime {
 public:
  ScopedRealtime() : prior_(MarkRealtime(true)) {}
  ~ScopedRealtime() {
    CHECK(MarkRealtime(prior_)) << ": Priority was modified";
  }

 private:
  const bool prior_;
};

// Class which marks us as not on the RT scheduler until it goes out of scope.
// Note: this shouldn't be needed for most applications.
class ScopedNotRealtime {
 public:
  ScopedNotRealtime() : prior_(MarkRealtime(false)) {}
  ~ScopedNotRealtime() {
    CHECK(!MarkRealtime(prior_)) << ": Priority was modified";
  }

 private:
  const bool prior_;
};

}  // namespace aos

#endif  // AOS_REALTIME_H_
