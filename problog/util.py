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

import logging
import time

import signal
import sys
import os
import subprocess
import distutils.spawn
import tempfile
import imp
import collections


def init_logger(verbose=None, name='problog'):
    """Initialize default logger.

    :param verbose: verbosity level (0: WARNING, 1: INFO, 2: DEBUG)
    :type verbose: int
    :param name: name of the logger (default: problog)
    :type name: str
    :return: result of ``logging.getLogger(name)''
    :rtype: logging.Logger
    """
    logger = logging.getLogger(name)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if not verbose:
        logger.setLevel(logging.WARNING)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
        logger.info('Output level: INFO')
    else:
        logger.setLevel(logging.DEBUG)
        logger.debug('Output level: DEBUG')
    return logger


class Timer(object):
    """Report timing information for a block of code.
    To be used as a ``with'' block.

    :param msg: message to print
    :type msg: str
    :param output: file object to write to (default: write to logger ``problog'')
    :type output: file
    """

    def __init__(self, msg, output=None):
        self.message = msg
        self.start_time = None
        self.output = output

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args):
        if self.output is None:
            logger = logging.getLogger('problog')
            logger.info('%s: %.4fs' % (self.message, time.time()-self.start_time))
        else:
            print ('%s: %.4fs' % (self.message, time.time()-self.start_time), file=self.output)


def _raise_timeout(*args):
    """Raise global timeout exception (used by global timer)

    :param args: signal information (ignored)
    :raise KeyboardInterrupt:
    """
    raise KeyboardInterrupt('Timeout')   # Global exception on all threads


def start_timer(timeout=0):
    """Start a global timeout timer.

    :param timeout: timeout in seconds
    :type timeout: int
    """
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(timeout)


def stop_timer():
    """Stop the global timeout timer."""
    signal.alarm(0)


def subprocess_check_call(*popenargs, **kwargs):
    """Wrapper for subprocess.check_call that recursively kills subprocesses when Python is \
    interrupted.

    Additionally expands executable name to full path.

    :param popenargs: positional arguments of subprocess.call
    :param kwargs: keyword arguments of subprocess.call
    :return: result of subprocess.call
    """
    retcode = subprocess_call(*popenargs, **kwargs)
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
        raise subprocess.CalledProcessError(retcode, cmd)
    return 0


def subprocess_call(*popenargs, **kwargs):
    """Wrapper for subprocess.call that recursively kills subprocesses when Python is interrupted.

    Additionally expands executable name to full path.

    :param popenargs: positional arguments of subprocess.call
    :param kwargs: keyword arguments of subprocess.call
    :return: result of subprocess.call
    """
    process = None
    try:
        popenargs = _find_process(*popenargs)
        process = subprocess.Popen(*popenargs, **kwargs)
        return process.wait()
    except KeyboardInterrupt:
        kill_proc_tree(process)
        raise
    except SystemExit:
        kill_proc_tree(process)
        raise


def _find_process(cmd, *rest):
    fullname = distutils.spawn.find_executable(cmd[0])
    if fullname is not None:
        return ([fullname] + cmd[1:],) + rest
    else:
        return (cmd,) + rest


def kill_proc_tree(process, including_parent=True):
    """Recursively kill a subprocess. Useful when the subprocess is a script.
    Requires psutil but silently fails when it is not present.

    :param process: process
    :type process: subprocess.Popen
    :param including_parent: also kill process itself (default: True)
    :type including_parent: bool
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
        psutil = None
        process.kill()


class OrderedSet(collections.MutableSet):
    """Provides an ordered version of a set which keeps elements in the order they are added.

    :param iterable: add elements from this iterable (default: None)
    :type iterable: Sequence
    """

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


def mktempfile(suffix=''):
    """Create a temporary file with the given name suffix.

    :param suffix: extension of the file
    :type suffix: str
    :return: name of the temporary file
    """
    fd, filename = tempfile.mkstemp(suffix)
    os.close(fd)
    return filename


def load_module(filename):
    """Load a Python module from a filename or qualified module name.

    If filename ends with ``.py'', the module is loaded from the given file.
    Otherwise it is taken to be a module name reachable from the path.

    Example:

    .. code-block: python

       pb_util = load_module('problog.util')
       pb_util = load_module('problog/util.py')

    :param filename: location of the module
    :type filename: str
    :return: loaded module
    :rtype: module
    """
    if filename.endswith('.py'):
        filename = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
        (path, name) = os.path.split(filename)
        (name, ext) = os.path.splitext(name)
        (extfile, filename, data) = imp.find_module(name, [path])
        return imp.load_module(name, extfile, filename, data)
    else:
        mod = __import__(filename)
        components = filename.split('.')
        for c in components[1:]:
            mod = getattr(mod, c)
        return mod


def format_value(data, precision=8):
    """Pretty print a given value.

    :param data: data to format
    :param precision: max. number of digits
    :type precision: int
    :return: pretty printed result
    :rtype: str
    """
    if isinstance(data, float):
        data = ('%.' + str(precision) + 'g') % data
    return ('{:<' + str(precision+2) + '}').format(data)


def format_tuple(data, precision=8, columnsep='\t'):
    """Pretty print a given tuple (or single value).

    :param data: data to format
    :param precision: max. number of digits
    :type precision: int
    :param columnsep: column separator
    :type columnsep: str
    :return: pretty printed result
    :rtype: str
    """

    if isinstance(data, str):
        # is a string -> return string itself
        return data
    elif isinstance(data, collections.Sequence):
        return columnsep.join(map(lambda v: format_value(v, precision=precision), data))
    else:
        return format_value(data, precision=precision)


def format_dictionary(data, precision=8, keysep=':', columnsep='\t'):
    """Pretty print a given dictionary.

    :param data: data to format
    :type data: dict
    :param precision: max. number of digits
    :type precision: int
    :param keysep: separator between key and value (default: ``;'')
    :type keysep: str
    :param columnsep: column separator (default: ``tab'')
    :type columnsep: str
    :return: pretty printed result
    :rtype: str
    """
    if not data:
        return ""  # no queries
    else:
        s = []
        # Determine maximum length of key
        l = max(len(str(k)) for k in data)
        fmt = ('{:>' + str(l) + '}{}{}{}')
        for it in sorted([(str(k), v) for k, v in data.items()]):
            val = format_tuple(it[1], precision=precision, columnsep=columnsep)
            s.append(fmt.format(it[0], keysep, columnsep, val))
        return '\n'.join(s)
