
#ifndef AED_COMMON_UTIL_TIMER_HPP
#define AED_COMMON_UTIL_TIMER_HPP

#include "common/network/packet.hpp"
#include <vector>
#include <boost/thread/mutex.hpp>
#include <stdint.h>
#include <sys/time.h>
#include <tr1/unordered_map>
#include <stdio.h>

namespace aed {
namespace common {
namespace util {

class Timer
{
public:
    Timer();                            // Resource Acquisition Is Initialization
    ~Timer();                           // default destructor, do nothing here

    void   start();                     // start timer
    void   stop();                      // stop the timer
    double getElapsedTime();            // get elapsed time in second
    double getElapsedTimeInSec();       // same as getElapsedTime
    double getElapsedTimeInMilliSec();  // get elapsed time in milli-second
    double getElapsedTimeInMicroSec();  // get elapsed time in micro-second

    int64_t getStartTimeInMicroSec();
    int64_t getEndTimeInMicroSec();

private:
    // stop flag. 
    // If this timer is stopped, then you get the same time with last valid 
    // timing from any timing function
    int    stopped;                     

#ifdef WIN32
    LARGE_INTEGER frequency;            // ticks per second
    LARGE_INTEGER startCount;           //
    LARGE_INTEGER endCount;             //
#else
    timeval startCount;                 //
    timeval endCount;                   //
#endif
};

// |--Frontend id--|---Frame num---|----Object z-sort index----|
// |    16 bits    |    16 bits    |          32 bits          |
#define MAKE_TIMER_ID(render_order,frame,fe_id) (((uint64_t) render_order) | (((uint64_t) frame) << 32) | (((uint64_t)fe_id) << 48))

typedef uint64_t TimerTypeHandle;
typedef uint64_t TrackedObjId;

extern TimerTypeHandle doNotTimeHandle;

struct ObjectTimePair
{
    TrackedObjId object;
    pthread_t threadId;
    double totalElapsedSeconds;
    int64_t startTime_us;
    int64_t endTime_us;
};

struct TimerInstanceData
{
    TimerTypeHandle type;
    ObjectTimePair timeData;
};

struct ProcessTimerData
{
	// Timer data for this process
    const char* processName;
    double totalElapsedSeconds;
    std::vector<ObjectTimePair>* timerInstances;
};

struct AllocatedMemoryData
{
	size_t size;
	TimerTypeHandle type;
};

// When a pointer is allocated or freed this is created
struct MemActivityInstance
{
    bool allocation;     // True if alloc, false if free
    timeval time; 
    size_t size;
};

struct ProcessMemoryData
{
    const char* processName;
    std::vector<MemActivityInstance>* memActivityInstances;
};


class TimerManager
{
public:
    static TimerManager& instance()
    {
        static TimerManager tm;
        return tm;
    }
    ~TimerManager();

	// XXX most "timer" things need to be renamed since they apply to time and memory now
    TimerTypeHandle registerTimer(const char* process_name);
    TimerTypeHandle registerMemTracker(const char* processName);
    void setCommentString(const char* comment_str);
    void addTimerData(TimerInstanceData &data);
    void writeTimerDataToFile(const char* timerFileName, bool verbose);
    void writeMemoryDataToFile(const char* memFileName);
    void writeTimeline(const char* timerFileName);
    ProcessTimerData getTimerData(TimerTypeHandle handle);
    TrackedObjId constructTrackedObjectId(objid_t ident, int frontend_id); 
	void addMemoryData(void* ptr, size_t size, TimerTypeHandle typeHandle);
	void freeMemory(void* ptr);

private:

    bool gatherTimingData;

    TimerManager() : gatherTimingData(false) {}

    char* comment_str;
    int64_t initialStartTime;
    std::vector<ProcessTimerData> processTimers;
    boost::mutex manager_mutex;
	
	size_t totalAllocatedMemory;
	std::tr1::unordered_map<void*, AllocatedMemoryData> allocatedPointers;

    std::vector<ProcessMemoryData> processMemoryTrackers;
};

#define TIMER_MANAGER_OBJ TimerManager::instance()

class ManagedTimer
{
public:
    ManagedTimer(TimerTypeHandle h, TrackedObjId objectId);
    ~ManagedTimer();

    void start();
    void stop();
    double getElapsedTimeInSec();
    void reportTimerData();

    bool fluid;
private:
    TimerTypeHandle typeHandle;
    TrackedObjId    trackedObject;
    pthread_t       threadId;

    bool reported;
    Timer internalTimer;
};

} // util
} // common
} // aed

#endif // TIMER_H_DEF

