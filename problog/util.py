from __future__ import print_function

import logging, time

import signal
import os
import subprocess
import sys
import distutils.spawn

class Timer(object) :

    def __init__(self, msg) :
        self.message = msg
        self.start_time = None
        
    def __enter__(self) :
        self.start_time = time.time()
        
    def __exit__(self, *args) :
        logger = logging.getLogger('problog')
        logger.info('%s: %.4fs' % (self.message, time.time()-self.start_time))

def raise_timeout(*args) :
    raise KeyboardInterrupt('Timeout')   # Global exception on all threads

def start_timer(timeout=0) :
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(timeout)
    
def stop_timer() :
    signal.alarm(0)
    
def subprocess_check_call(*popenargs, **kwargs) :
    retcode = subprocess_call(*popenargs, **kwargs)
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd)
    return 0
   
def find_process(cmd, *rest) :
    cmd[0] = distutils.spawn.find_executable(cmd[0])
    return (cmd,) + rest
            
def subprocess_call(*popenargs, **kwargs):
    try :
        popenargs = find_process(*popenargs)
        process = subprocess.Popen(*popenargs, **kwargs)
        return process.wait()
    except KeyboardInterrupt :
        process.kill()
        raise
