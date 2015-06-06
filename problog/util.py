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
        
import collections

class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = [] 
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:        
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if other is None :
            return False
        elif isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

