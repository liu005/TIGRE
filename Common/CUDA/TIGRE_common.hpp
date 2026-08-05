#ifndef _COMMON_HPP_20201017_
#define _COMMON_HPP_20201017_

#define STRINGIFY(n) #n
#define TOSTRING(n) STRINGIFY(n)
#define __HERE__ __FILE__ " (" TOSTRING(__LINE__) "): "
#define PRINT_HERE printf(__HERE__);printf
// #define PRINT_HERE (void*)0

#if defined(IS_FOR_PYTIGRE)
#ifndef IS_FOR_MATLAB_TIGRE
    #define IS_FOR_MATLAB_TIGRE 0
#endif  // IS_FOR_MATLAB_TIGRE
void mexPrintf(const char*, ...);
void mexErrMsgIdAndTxt(const char* pcTag, const char* pcMsg);
void mexWarnMsgIdAndTxt(const char* pcTag, const char* pcMsg);
#else
#ifndef IS_FOR_MATLAB_TIGRE
    #define IS_FOR_MATLAB_TIGRE 1
#endif  // IS_FOR_MATLAB_TIGRE
#include "mex.h"
#include "tmwtypes.h"
#endif  // IS_TIGRE_FOR_PYTHON

/* Last error recorded by cudaCheckErrors(), so a binding can report WHAT
 * failed rather than only that something did.
 *
 * Under IS_FOR_PYTIGRE, mexErrMsgIdAndTxt() used to exit(1): any CUDA error
 * terminated the host Python process - no exception, no traceback, nothing the
 * caller could catch, and the interpreter gone along with any unsaved work. A
 * library must not end its host process. Under MATLAB the same call longjmps
 * out of the middle of a CUDA function, running no cleanup, so every error
 * leaked every device buffer, page-locked allocation, stream and texture held
 * at that moment for the rest of the session.
 *
 * The CUDA entry points now return the codes from errors.hpp instead and clean
 * up on the way out (see tigre_cleanup.hpp). Each binding reports at its own
 * boundary: MATLAB raises a MATLAB error, and the Cython layer raises a
 * TigreCudaCallError - the mechanism its error_list was always indexed against.
 *
 * Declared for both bindings so the CUDA sources stay binding-agnostic.
 */
void tigreSetLastError(const char* pcTag, const char* pcMsg);
const char* tigreGetLastError(void);
void tigreClearLastError(void);

#endif  // _COMMON_HPP_20201017_
