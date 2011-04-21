#include "cuda_err.hh"

#include <stdio.h>

void exit_on_err(cudaError_t err) {
	char * err_str;
	switch (err) {
		case cudaSuccess                      : return;
		case cudaErrorMissingConfiguration    : err_str = "Missing configuration error."      ; break;
		case cudaErrorMemoryAllocation        : err_str = "Memory allocation error."          ; break;
		case cudaErrorInitializationError     : err_str = "Initialization error."             ; break;
		case cudaErrorLaunchFailure           : err_str = "Launch failure."                   ; break;
		case cudaErrorPriorLaunchFailure      : err_str = "Prior launch failure."             ; break;
		case cudaErrorLaunchTimeout           : err_str = "Launch timeout error."             ; break;
		case cudaErrorLaunchOutOfResources    : err_str = "Launch out of resources error."    ; break;
		case cudaErrorInvalidDeviceFunction   : err_str = "Invalid device function."          ; break;
		case cudaErrorInvalidConfiguration    : err_str = "Invalid configuration."            ; break;
		case cudaErrorInvalidDevice           : err_str = "Invalid device."                   ; break;
		case cudaErrorInvalidValue            : err_str = "Invalid value."                    ; break;
		case cudaErrorInvalidPitchValue       : err_str = "Invalid pitch value."              ; break;
		case cudaErrorInvalidSymbol           : err_str = "Invalid symbol."                   ; break;
		case cudaErrorMapBufferObjectFailed   : err_str = "Map buffer object failed."         ; break;
		case cudaErrorUnmapBufferObjectFailed : err_str = "Unmap buffer object failed."       ; break;
		case cudaErrorInvalidHostPointer      : err_str = "Invalid host pointer."             ; break;
		case cudaErrorInvalidDevicePointer    : err_str = "Invalid device pointer."           ; break;
		case cudaErrorInvalidTexture          : err_str = "Invalid texture."                  ; break;
		case cudaErrorInvalidTextureBinding   : err_str = "Invalid texture binding."          ; break;
		case cudaErrorInvalidChannelDescriptor: err_str = "Invalid channel descriptor."       ; break;
		case cudaErrorInvalidMemcpyDirection  : err_str = "Invalid memcpy direction."         ; break;
		case cudaErrorAddressOfConstant       : err_str = "Address of constant error."        ; break;
		case cudaErrorTextureFetchFailed      : err_str = "Texture fetch failed."             ; break;
		case cudaErrorTextureNotBound         : err_str = "Texture not bound error."          ; break;
		case cudaErrorSynchronizationError    : err_str = "Synchronization error."            ; break;
		case cudaErrorInvalidFilterSetting    : err_str = "Invalid filter setting."           ; break;
		case cudaErrorInvalidNormSetting      : err_str = "Invalid norm setting."             ; break;
		case cudaErrorMixedDeviceExecution    : err_str = "Mixed device execution."           ; break;
		case cudaErrorCudartUnloading         : err_str = "CUDA runtime unloading."           ; break;
		case cudaErrorUnknown                 : err_str = "Unknown error condition."          ; break;
		case cudaErrorNotYetImplemented       : err_str = "Function not yet implemented."     ; break;
		case cudaErrorMemoryValueTooLarge     : err_str = "Memory value too large."           ; break;
		case cudaErrorInvalidResourceHandle   : err_str = "Invalid resource handle."          ; break;
		case cudaErrorNotReady                : err_str = "Not ready error."                  ; break;
		case cudaErrorInsufficientDriver      : err_str = "CUDA runtime is newer than driver."; break;
		case cudaErrorSetOnActiveProcess      : err_str = "Set on active process error."      ; break;
		case cudaErrorNoDevice                : err_str = "No available CUDA device."         ; break;
		case cudaErrorStartupFailure          : err_str = "Startup failure."                  ; break;
		case cudaErrorApiFailureBase          : err_str = "API failure base."                 ; break;
	}
	printf("Error: %s\n", err_str);
	exit(-1);
}
