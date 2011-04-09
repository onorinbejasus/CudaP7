//////////////////////////////////////////////////////////////////////////////
// Timer.cpp
// =========
// High Resolution Timer.
// This timer is able to measure the elapsed time with 1 micro-second accuracy
// in both Windows, Linux and Unix system 
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2003-01-13
// UPDATED: 2006-01-13
//
// Copyright (c) 2003 Song Ho Ahn
//////////////////////////////////////////////////////////////////////////////

#include "timer.hpp"

namespace aed {
namespace common {
namespace util {

///////////////////////////////////////////////////////////////////////////////
// constructor
///////////////////////////////////////////////////////////////////////////////
Timer::Timer()
{
#ifdef WIN32
    QueryPerformanceFrequency(&frequency);
    startCount.QuadPart = 0;
    endCount.QuadPart = 0;
#else
    startCount.tv_sec = startCount.tv_usec = 0;
    endCount.tv_sec = endCount.tv_usec = 0;
#endif

    start();
}



///////////////////////////////////////////////////////////////////////////////
// distructor
///////////////////////////////////////////////////////////////////////////////
Timer::~Timer()
{
}



///////////////////////////////////////////////////////////////////////////////
// start timer.
// startCount will be set at this point.
///////////////////////////////////////////////////////////////////////////////
void Timer::start()
{
    stopped = 0; // reset stop flag
#ifdef WIN32
    QueryPerformanceCounter(&startCount);
#else
    gettimeofday(&startCount, NULL);
#endif
}


///////////////////////////////////////////////////////////////////////////////
// stop the timer.
// endCount will be set at this point.
///////////////////////////////////////////////////////////////////////////////
void Timer::stop()
{
    stopped = 1; // set timer stopped flag

#ifdef WIN32
    QueryPerformanceCounter(&endCount);
#else
    gettimeofday(&endCount, NULL);
#endif
}



///////////////////////////////////////////////////////////////////////////////
// compute elapsed time in micro-second resolution.
// other getElapsedTime will call this first, then convert to correspond resolution.
///////////////////////////////////////////////////////////////////////////////
double Timer::getElapsedTimeInMicroSec()
{
#ifdef WIN32
    if(!stopped)
        QueryPerformanceCounter(&endCount);

    double startTimeInMicroSec = startCount.QuadPart * (1000000.0 / frequency.QuadPart);
    double endTimeInMicroSec = endCount.QuadPart * (1000000.0 / frequency.QuadPart);
#else
    if(!stopped)
        gettimeofday(&endCount, NULL);

    double startTimeInMicroSec = (startCount.tv_sec * 1000000.0) + startCount.tv_usec;
    double endTimeInMicroSec = (endCount.tv_sec * 1000000.0) + endCount.tv_usec;
#endif

    return endTimeInMicroSec - startTimeInMicroSec;
}



///////////////////////////////////////////////////////////////////////////////
// divide elapsedTimeInMicroSec by 1000
///////////////////////////////////////////////////////////////////////////////
double Timer::getElapsedTimeInMilliSec()
{
    return this->getElapsedTimeInMicroSec() * 0.001;
}



///////////////////////////////////////////////////////////////////////////////
// divide elapsedTimeInMicroSec by 1000000
///////////////////////////////////////////////////////////////////////////////
double Timer::getElapsedTimeInSec()
{
    return this->getElapsedTimeInMicroSec() * 0.000001;
}



///////////////////////////////////////////////////////////////////////////////
// same as getElapsedTimeInSec()
///////////////////////////////////////////////////////////////////////////////
double Timer::getElapsedTime()
{
    return this->getElapsedTimeInSec();
}

int64_t Timer::getStartTimeInMicroSec()
{
    return ((int64_t)startCount.tv_sec * 1000000L) + (int64_t)startCount.tv_usec;
}

int64_t Timer::getEndTimeInMicroSec()
{
    return ((int64_t)endCount.tv_sec * 1000000L) + (int64_t)endCount.tv_usec;
}


TimerTypeHandle doNotTimeHandle = 0;

TimerTypeHandle TimerManager::registerTimer(const char* processName)
{
    if(!gatherTimingData)
    {
        return doNotTimeHandle;
    }

    boost::mutex::scoped_lock lock(manager_mutex);
    
    ProcessTimerData newData;
    
    newData.processName = (const char*)new char[strlen(processName)+1];
    strncpy((char*)newData.processName, processName, strlen(processName)+1);
    
    newData.totalElapsedSeconds = 0;
    newData.timerInstances = new std::vector<ObjectTimePair>();

    processTimers.push_back(newData);

    // handle is currently the index in the processTimers vector + 1
    return processTimers.size();
}

// Works the same as register timer, adds to the memory tracker vector
TimerTypeHandle TimerManager::registerMemTracker(const char* processName)
{
    if(!gatherTimingData)
    {
        return doNotTimeHandle;
    }

    boost::mutex::scoped_lock lock(manager_mutex);
    
    ProcessMemoryData memoryData;
    memoryData.processName = (const char*)new char[strlen(processName)+1];
    strncpy((char*)memoryData.processName, processName, strlen(processName)+1);

    memoryData.memActivityInstances = new std::vector<MemActivityInstance>();
    printf("memActivityInstances: %p\n", memoryData.memActivityInstances);

    processMemoryTrackers.push_back(memoryData);

    // handle is currently the index in the processMemoryTrackers vector + 1
    return processMemoryTrackers.size();
}

void TimerManager::addTimerData(TimerInstanceData &data)
{
    if(!gatherTimingData) return;

    boost::mutex::scoped_lock lock(manager_mutex);

    // add data to vector
    processTimers[data.type-1].totalElapsedSeconds += data.timeData.totalElapsedSeconds;
    processTimers[data.type-1].timerInstances->push_back(data.timeData);
}

void TimerManager::addMemoryData(void* ptr, size_t size, TimerTypeHandle type)
{
    if(!gatherTimingData) return;

    boost::mutex::scoped_lock lock(manager_mutex);

    // Add a MemActivityInstance for this allocation
    MemActivityInstance mai;
    mai.allocation = true;
    mai.size = size;
    gettimeofday(&mai.time, NULL);
    processMemoryTrackers[type-1].memActivityInstances->push_back(mai);

    size_t s = processMemoryTrackers[type-1].memActivityInstances->size();

    // Keep track of the pointer
	AllocatedMemoryData amd;
	amd.size = size;
	amd.type = type;
	allocatedPointers[ptr] = amd;
	totalAllocatedMemory += size;
}

void TimerManager::freeMemory(void* ptr)
{
    if(!gatherTimingData) return;

    boost::mutex::scoped_lock lock(manager_mutex);

    // Add a MemActivityInstance for this allocation
    MemActivityInstance mai;
    mai.allocation = false;
    mai.size = allocatedPointers[ptr].size;
    gettimeofday(&mai.time, NULL);
    processMemoryTrackers[allocatedPointers[ptr].type-1].memActivityInstances->push_back(mai);

    // Keep track of the pointer
	totalAllocatedMemory -= allocatedPointers[ptr].size;
	allocatedPointers.erase(ptr);
}



void TimerManager::setCommentString(const char* comment)
{
    if(!gatherTimingData) return;

    if(comment == NULL) {
        comment_str = NULL;
        return;
    }
    int max_len = 100;
    comment_str = new char[max_len];
    strncpy(comment_str, comment, max_len);
}

void TimerManager::writeTimerDataToFile(const char* timerFileName, bool verbose = true)
{
    if(!gatherTimingData) {
        return;
    }
    
    // write timer data to a file
    FILE* timerFile = fopen(timerFileName, "w");

    if(comment_str != NULL) {
        fprintf(timerFile, "Comment: %s\n", comment_str);
    }

    for(size_t i = 0; i < processTimers.size(); ++i)
    {
        std::vector<ObjectTimePair>* currentInstances = processTimers[i].timerInstances;
        int64_t firstStartTime = 0, lastEndTime = 0;
        double totalProgramTime = 0, utilization = 0;
        if(currentInstances->size() != 0)
        {
            firstStartTime = (*currentInstances)[0].startTime_us;
            lastEndTime = (*currentInstances)[currentInstances->size()-1].endTime_us;
            totalProgramTime = (lastEndTime - firstStartTime) / 1000000.0;
            utilization = processTimers[i].totalElapsedSeconds / totalProgramTime;
        }

        unsigned int currentSize = currentInstances->size();

        if(verbose)
        {
            fprintf(timerFile, "Process name: %s\nNumTimesCalled: %u\nTotalElapsedSeconds: %f\nUtilization: %f\n",
                    processTimers[i].processName,
                    currentSize,
                    processTimers[i].totalElapsedSeconds,
                    utilization
                    );
        }
        else 
        {
            fprintf(timerFile, "%s\n%u\n%f\n%f\n",
                    processTimers[i].processName,
                    currentSize,
                    processTimers[i].totalElapsedSeconds,
                    utilization
                    );
        }


        if(verbose)
            fprintf(timerFile, "\tObject id\tThread id\tElapsedSeconds\tStart time\tEnd time:\n");

        for(size_t j = 0; j < currentSize; j++)
        {
            if(verbose)
                fprintf(timerFile, "\t");

            fprintf(timerFile, "%ld\t%ld\t%f\t%ld\t%ld\n", 
                    (*currentInstances)[j].object, 
                    (*currentInstances)[j].threadId,
                    (*currentInstances)[j].totalElapsedSeconds,
                    (*currentInstances)[j].startTime_us - initialStartTime,
                    (*currentInstances)[j].endTime_us - initialStartTime
                    );
        }

        fprintf(timerFile, "\n");
    }
    
    fclose(timerFile);
}

void TimerManager::writeMemoryDataToFile(const char* memFileName)
{
    if(!gatherTimingData) {
        return;
    }
    
    // Write memory data to a file
    FILE* memFile = fopen(memFileName, "w");

    if(comment_str != NULL) {
        fprintf(memFile, "Comment: %s\n", comment_str);
    }

    // Write the list of allocs and frees made by each stage
    for(size_t i = 0; i < processMemoryTrackers.size(); i++)
    {
        std::vector<MemActivityInstance>* currentInstances = processMemoryTrackers[i].memActivityInstances;        
        fprintf(memFile, "%s\n%zu\n", processMemoryTrackers[i].processName, currentInstances->size());

        unsigned int currentSize = currentInstances->size();
        for(size_t j = 0; j < currentSize; j++)
        {
            fprintf(memFile, "%d\t%f\t%ld\n",
                    (*currentInstances)[j].allocation,
                    ((*currentInstances)[j].time.tv_sec * 1000000.0) + ((*currentInstances)[j].time.tv_usec),
                    (*currentInstances)[j].size
                   );
        }

        fprintf(memFile, "\n");
    }

    fclose(memFile);
}


TimerManager::~TimerManager()
{
    if(!gatherTimingData) return;
    writeTimerDataToFile("timerData.txt");
}

ProcessTimerData TimerManager::getTimerData(TimerTypeHandle handle)
{
	// XXX this "handle-1" business should at least be a macro
    return processTimers[handle-1];
}




// Layout of a TrackedObjId:
// |--Frontend id--|---Frame num---|----Object z-sort index----|
// |    16 bits    |    16 bits    |          32 bits          |
//
// XXX don't need this function anymore - analyzer makes the ID and it carries
// all the way through the pipeline
/*
TrackedObjId TimerManager::constructTrackedObjectId(objid_t ident, int frontend_id)
{
    // Assumes only lower 48 bits of ident are used. The rest can identify frontend.
    
    // (frontend id cannot be larger than 16 bits!)
    assert(frontend_id < (1 << 16));

    int64_t f64 = frontend_id;
    return ident | (f64 << 48);
}
*/

ManagedTimer::ManagedTimer(TimerTypeHandle h, TrackedObjId object = -1) 
    : fluid(false), typeHandle(h), trackedObject(object), threadId(pthread_self()), reported(false) 
{
    // start upon initialization
    // Internal timer starts automatically
}

ManagedTimer::~ManagedTimer() 
{
    //if(fluid)
    //    printf("Destroying a timer\n");

    // stop and report upon destruction
    internalTimer.stop();
    reportTimerData();
}

void ManagedTimer::start()
{
    internalTimer.start();
}

void ManagedTimer::stop()
{
    internalTimer.stop();
}

double ManagedTimer::getElapsedTimeInSec()
{
    return internalTimer.getElapsedTimeInSec();
}

void ManagedTimer::reportTimerData()
{
    if(typeHandle == doNotTimeHandle || reported) {
        return;
    }

    TimerInstanceData data;
    data.type = typeHandle;
    data.timeData.threadId = threadId;
    data.timeData.object = trackedObject;
    data.timeData.totalElapsedSeconds = internalTimer.getElapsedTimeInSec();
    data.timeData.startTime_us = internalTimer.getStartTimeInMicroSec();
    data.timeData.endTime_us = internalTimer.getEndTimeInMicroSec();

    TIMER_MANAGER_OBJ.addTimerData(data);

    reported = true;
}

} // util
} // common
} // aed

