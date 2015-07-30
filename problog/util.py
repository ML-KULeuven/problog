"""
Part of the ProbLog distribution.

Copyright 2015 KU Leuven, DTAI Research Group

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import print_function

import logging, time

import signal
import os
import subprocess
import sys
import distutils.spawn

class Timer(object) :

    def __init__(self, msg, output=None) :
        self.message = msg
        self.start_time = None
        self.output = output

    def __enter__(self) :
        self.start_time = time.time()

    def __exit__(self, *args) :
        if self.output is None:
            logger = logging.getLogger('problog')
            logger.info('%s: %.4fs' % (self.message, time.time()-self.start_time))
        else:
            print ('%s: %.4fs' % (self.message, time.time()-self.start_time), file=self.output)

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
    except KeyboardInterrupt:
        kill_proc_tree(process)
        raise
    except SystemExit:
        kill_proc_tree(process)
        raise


def kill_proc_tree(process, including_parent=True):
    """
    Recursively kill a subprocess. Useful when the subprocess is a script.
    Requires psutil but silently fails when it is not present.
    """
    try:
        import psutil
        pid = process.pid
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        psutil.wait_procs(children, timeout=5)
        if including_parent:
            parent.kill()
            parent.wait(5)
    except ImportError:
        process.kill()

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
