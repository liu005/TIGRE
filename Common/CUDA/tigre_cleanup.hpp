/*-------------------------------------------------------------------------
 *
 * Scope-based cleanup for the CUDA code.
 *
 * WHY THIS EXISTS
 *
 * The CUDA functions acquire a lot of resources - device buffers, page-locked
 * host memory, host registrations, streams, texture objects and plain mallocs -
 * and release them in one block at the end of the function. That is correct
 * only while nothing leaves the function early, which is exactly what error
 * handling needs to do. Historically neither binding managed it:
 *
 *   - MATLAB: cudaCheckErrors() called mexErrMsgIdAndTxt(), which longjmps back
 *     to MATLAB from the middle of the function. A longjmp runs no destructors
 *     and no cleanup code, so every CUDA error leaked every resource held at
 *     that moment, for the lifetime of the MATLAB session.
 *   - Python: mexErrMsgIdAndTxt() called exit(1), terminating the host process.
 *
 * Registering a release action next to each acquisition makes an early
 * `return ERR_CUDA` safe, so the error can be reported at the binding boundary
 * (a MATLAB error, or a Python TigreCudaCallError) instead of unwinding through
 * live resources or killing the process.
 *
 * Actions run in reverse order of registration, so resources are released in
 * the opposite order to acquisition. Releases are best effort and never throw:
 * during error handling the CUDA context may already be unusable, and a failure
 * to free must not mask the original error.
 *
 * This header uses no CUDA features beyond the runtime API calls the callers
 * already make, so it does not change which GPUs or CUDA versions are
 * supported.
 *
 * ---------------------------------------------------------------------------
 * This file is part of the TIGRE Toolbox
 * License:  Open Source under BSD.
 *           See the full license at
 *           https://github.com/CERN/TIGRE/blob/master/LICENSE
 * ---------------------------------------------------------------------------
 */

#ifndef TIGRE_CLEANUP_HPP
#define TIGRE_CLEANUP_HPP

#include <cuda_runtime_api.h>

#include <functional>
#include <utility>
#include <vector>

namespace tigre {

/** Collects release actions and runs them when it goes out of scope. */
class CleanupScope {
public:
    CleanupScope() {}
    ~CleanupScope() { run(); }

    /** Register a release action. Called in reverse order on scope exit. */
    void add(std::function<void()> action) {
        actions_.push_back(std::move(action));
    }

    /** Release everything now; safe to call more than once. */
    void run() {
        for (std::vector<std::function<void()> >::reverse_iterator it = actions_.rbegin();
             it != actions_.rend(); ++it) {
            // Best effort: a failing release during error handling must not
            // prevent the remaining ones, nor replace the original error.
            try {
                (*it)();
            } catch (...) {
            }
        }
        actions_.clear();
        // Discard any error raised BY the releases themselves.
        //
        // Cleaning up must not change the error state the next call observes.
        // These run while unwinding a failure, so they routinely operate on
        // half-built state - unregistering a buffer that never registered,
        // freeing a pointer whose allocation failed - and each of those leaves
        // an error pending. cudaGetLastError() is sticky-until-read, so the
        // NEXT entry point's first check would read it and report a failure
        // that belongs to the previous call, making the library appear broken
        // from then on. The error being reported has already been captured by
        // the caller, so dropping these is safe.
        cudaGetLastError();
    }

    /** Abandon the registered actions (the caller has taken ownership). */
    void dismiss() { actions_.clear(); }

private:
    CleanupScope(const CleanupScope&);
    CleanupScope& operator=(const CleanupScope&);

    std::vector<std::function<void()> > actions_;
};

}  // namespace tigre

#endif  // TIGRE_CLEANUP_HPP
