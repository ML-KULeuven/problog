/* File: sdd.i */
%module sdd


%{
#define SWIG_FILE_WITH_INIT
#include "sddapi.h"
#include "parameters.h"
#include "compiler.h"
#include "array_access.h"

#include <execinfo.h>
#include <signal.h>

#include "except.h"

void c_handler(int sig)
{
  throw(sig);
}

%}

%include exception.i

%exception {
//
//    signal(SIGFPE,  handler);
//    signal(SIGILL,  handler);

// signal(SIGSEGV, handler);


    void (*h_alrm)(int);
    void (*h_abrt)(int);
    void (*h_term)(int);
    void (*h_int)(int);
    h_alrm = signal(SIGALRM,  c_handler);
    h_abrt = signal(SIGABRT, c_handler);
    h_term = signal(SIGTERM, c_handler);
    h_int = signal(SIGINT,  c_handler);
    try {
        $action
    } catch(SIGINT) {
        SWIG_exception(SWIG_SystemError, "Process interrupted");
    } catch(SIGTERM) {
        SWIG_exception(SWIG_SystemError, "Process terminated");
    } catch(SIGABRT) {
        SWIG_exception(SWIG_SystemError, "Process aborted");
    } catch(SIGALRM) {
        SWIG_exception(SWIG_SystemError, "Process timeout");
    }
    signal(SIGTERM, h_term);
    signal(SIGINT,  h_int);
    signal(SIGABRT, h_abrt);
    signal(SIGALRM, h_alrm);

}

%include "array_access.h"
%include "sddapi.h"
