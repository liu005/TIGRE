#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include "TIGRE_common.hpp"

#if defined(IS_FOR_PYTIGRE)
void mexPrintf(const char* format, ...) {
    PRINT_HERE("");
    va_list argpointer;
    va_start(argpointer, format);
    vprintf(format, argpointer);
    va_end(argpointer);
}
void mexErrMsgIdAndTxt(const char* pcTag, const char* pcMsg) {
    /* Record and return. This used to exit(1), which ended the host Python
     * process on any CUDA error - no exception, no traceback, nothing the
     * caller could catch. Callers now return the errors.hpp codes and the
     * Cython layer raises TigreCudaCallError, so an error is something the
     * caller can handle rather than the end of the interpreter. */
    PRINT_HERE("%s %s\n", pcTag, pcMsg);
    tigreSetLastError(pcTag, pcMsg);
}
void mexWarnMsgIdAndTxt(const char* pcTag, const char* pcMsg) {
    PRINT_HERE("%s %s\n", pcTag, pcMsg);
}
#endif  // IS_FOR_PYTIGRE

/* Detail of the last recorded error, shared by both bindings.
 *
 * One static buffer suffices: the CUDA entry points are called one at a time
 * from the host language, and the binding reads the message immediately after
 * a non-zero return. */
static char s_lastError[512] = {0};

void tigreSetLastError(const char* pcTag, const char* pcMsg) {
    const char* tag = pcTag ? pcTag : "";
    const char* msg = pcMsg ? pcMsg : "";
#if defined(_MSC_VER)
    _snprintf_s(s_lastError, sizeof(s_lastError), _TRUNCATE, "%s %s", tag, msg);
#else
    snprintf(s_lastError, sizeof(s_lastError), "%s %s", tag, msg);
#endif
}

const char* tigreGetLastError(void) {
    return s_lastError;
}

void tigreClearLastError(void) {
    s_lastError[0] = '\0';
}
