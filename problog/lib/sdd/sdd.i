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

static void c_handler(int sig)
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

    struct sigaction new_action;
    new_action.sa_handler = c_handler;
    sigemptyset (&new_action.sa_mask);
    new_action.sa_flags = SA_RESTART;

    struct sigaction old_action_term;
    struct sigaction old_action_abrt;
    struct sigaction old_action_int;
    struct sigaction old_action_alrm;

    sigaction(SIGTERM,  &new_action, &old_action_term);
    sigaction(SIGABRT,  &new_action, &old_action_abrt);
    sigaction(SIGINT,  &new_action, &old_action_int);
    sigaction(SIGALRM,  &new_action, &old_action_alrm);
    try {
        $action
    } catch(SIGINT) {
        sigaction(SIGTERM, &old_action_term, NULL);
        sigaction(SIGABRT, &old_action_abrt, NULL);
        sigaction(SIGINT, &old_action_int, NULL);
        sigaction(SIGALRM, &old_action_alrm, NULL);
        SWIG_exception(SWIG_SystemError, "Process interrupted");
    } catch(SIGTERM) {
        sigaction(SIGTERM, &old_action_term, NULL);
        sigaction(SIGABRT, &old_action_abrt, NULL);
        sigaction(SIGINT, &old_action_int, NULL);
        sigaction(SIGALRM, &old_action_alrm, NULL);
        SWIG_exception(SWIG_SystemError, "Process terminated");
    } catch(SIGABRT) {
        sigaction(SIGTERM, &old_action_term, NULL);
        sigaction(SIGABRT, &old_action_abrt, NULL);
        sigaction(SIGINT, &old_action_int, NULL);
        sigaction(SIGALRM, &old_action_alrm, NULL);
        SWIG_exception(SWIG_SystemError, "Process aborted");
    } catch(SIGALRM) {
        sigaction(SIGTERM, &old_action_term, NULL);
        sigaction(SIGABRT, &old_action_abrt, NULL);
        sigaction(SIGINT, &old_action_int, NULL);
        sigaction(SIGALRM, &old_action_alrm, NULL);
        SWIG_exception(SWIG_SystemError, "Process timeout");
    }
    sigaction(SIGTERM, &old_action_term, NULL);
    sigaction(SIGABRT, &old_action_abrt, NULL);
    sigaction(SIGINT, &old_action_int, NULL);
    sigaction(SIGALRM, &old_action_alrm, NULL);

}

%include "array_access.h"
%include "sddapi.h"
