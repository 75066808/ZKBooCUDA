#include "cuda_runtime.h"

class Timer {
private:
	cudaEvent_t eventStart;
	cudaEvent_t eventStop;
	float time;

public:
	Timer() {
		cudaEventCreate(&eventStart);
		cudaEventCreate(&eventStop);
	}

	~Timer() {
		cudaEventDestroy(eventStart);
		cudaEventDestroy(eventStop);
	}

	void start(void) {
		cudaEventRecord(eventStart, 0);
		cudaEventSynchronize(eventStart);
	}

	void stop(void) {
		cudaEventRecord(eventStop, 0);
		cudaEventSynchronize(eventStop);
		cudaEventElapsedTime(&time, eventStart, eventStop);
	}

	float elapsed(void) {		
		return time;
	}
};