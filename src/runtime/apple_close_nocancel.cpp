#if defined(__APPLE__)

#include <cerrno>
#include <sys/cdefs.h>
#include <unistd.h>

// Darwin close() can report EINTR with ambiguous descriptor ownership. Use the
// non-cancellable entry point and treat its post-close EINTR/EINPROGRESS as
// success so cleanup never retries a descriptor that may already be reused.
#if !__DARWIN_NON_CANCELABLE
extern "C" {

#if !__DARWIN_ONLY_UNIX_CONFORMANCE
int cosmo_nbody_close_nocancel(int descriptor)
    __asm__("_close$NOCANCEL$UNIX2003");
#else
int cosmo_nbody_close_nocancel(int descriptor)
    __asm__("_close$NOCANCEL");
#endif

int close(int descriptor) {
    const int status = cosmo_nbody_close_nocancel(descriptor);
    if (status == -1 && (errno == EINTR || errno == EINPROGRESS)) {
        return 0;
    }
    return status;
}

} // extern "C"
#endif

#endif
