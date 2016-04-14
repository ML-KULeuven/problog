"""
problog.util - Useful utilities
-------------------------------

Provides useful utilities functions and classes.

..
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


class ProbLogLogFormatter(logging.Formatter):

    def __init__(self):
        logging.Formatter.__init__(self)

    def format(self, message):
        msg = str(message.msg) % message.args
        lines = msg.split('\n')
        if message.levelno < 10:
            linestart = '[LVL%s] ' % message.levelno
        else:
            linestart = '[%s] ' % message.levelname
        return linestart + ('\n' + linestart).join(lines)


def init_logger(verbose=None, name='problog', out=None):
    """Initialize default logger.

    :param verbose: verbosity level (0: WARNING, 1: INFO, 2: DEBUG)
    :type verbose: int
    :param name: name of the logger (default: problog)
    :type name: str
    :return: result of ``logging.getLogger(name)``
    :rtype: logging.Logger
    """
    if out is None:
        out = sys.stdout

    logger = logging.getLogger(name)
    ch = logging.StreamHandler(out)
    # formatter = logging.Formatter('[%(levelname)s] %(message)s')
    formatter = ProbLogLogFormatter()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if not verbose:
        logger.setLevel(logging.WARNING)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
        logger.info('Output level: INFO')
    elif verbose == 2:
        logger.setLevel(logging.DEBUG)
        logger.debug('Output level: DEBUG')
    else:
        level = max(1, 12 - verbose)   # between 9 and 1
        logger.setLevel(level)
        logger.log(level, 'Output level: %s' % level)
    return logger


class Timer(object):
    """Report timing information for a block of code.
    To be used as a ``with`` block.

    :param msg: message to print
    :type msg: str
    :param output: file object to write to (default: write to logger ``problog``)
    :type output: file
    """

    def __init__(self, msg, output=None, logger='problog'):
        self.message = msg
        self.start_time = None
        self.output = output
        self.logger = logger

    def __enter__(self):
        self.start_time = time.time()

    # noinspection PyUnusedLocal
    def __exit__(self, *args):
        if self.output is None:
            logger = logging.getLogger(self.logger)
            logger.info('%s: %.4fs' % (self.message, time.time() - self.start_time))
        else:
            print ('%s: %.4fs' % (self.message, time.time() - self.start_time), file=self.output)


# noinspection PyUnusedLocal
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


def subprocess_check_output(*popenargs, **kwargs):
    """Wrapper for subprocess.check_output that recursively kills subprocesses when Python is \
    interrupted.

    Additionally expands executable name to full path.

    :param popenargs: positional arguments of subprocess.call
    :param kwargs: keyword arguments of subprocess.call
    :return: result of subprocess.call
    """
    popenargs = _find_process(*popenargs)
    process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
    try:
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            raise subprocess.CalledProcessError(retcode, cmd, output=output)
        return output.decode()
    except KeyboardInterrupt:
        kill_proc_tree(process)
        raise
    except SystemExit:
        kill_proc_tree(process)
        raise


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
        # noinspection PyPackageRequirements
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
        # noinspection PyUnusedLocal
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
            # noinspection PyMethodFirstArgAssignment
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        """Add element.

        :param key: element to add
        """
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        """Discard element.

        :param key: element to remove
        """
        if key in self.map:
            key, prv, nxt = self.map.pop(key)
            prv[2] = nxt
            nxt[1] = prv

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
        """Remove and return first or last element.

        :param last: remove last element
        :return: last element
        """
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
        if other is None:
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

    If filename ends with ``.py``, the module is loaded from the given file.
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
    else:
        data = str(data)
    return ('{:<' + str(precision + 2) + '}').format(data)


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
        values = list(map(lambda v: format_value(v, precision=precision), data))
        if len(values) == 2 and values[0] == values[1]:
            values = [values[0]]
        return columnsep.join(values)
    else:
        return format_value(data, precision=precision)


def format_dictionary(data, precision=8, keysep=':', columnsep='\t'):
    """Pretty print a given dictionary.

    :param data: data to format
    :type data: dict
    :param precision: max. number of digits
    :type precision: int
    :param keysep: separator between key and value (default: ``;``)
    :type keysep: str
    :param columnsep: column separator (default: ``tab``)
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


class UHeap(object):
    """Updatable heap.

    Each element is represented as a pair (key, item).
    The operation ``pop()`` always returns the item with the smallest key.
    The operation ``push(item)`` either adds item (returns True) or updates its key (return False)
    A function for computing an item's key can be passed.

    :param key: function for computing the sort key of an item
    """

    def __init__(self, key=None):
        self._heap = []
        self._index = {}
        self._key = key

    def _compute_key(self, item):
        if self._key is None:
            return item
        else:
            return self._key(item)

    def push(self, item):
        """Add the item or update it's key in case it already exists.

        :param item: item to add
        :return: True is item was not in the collection
        """
        # Compute the item's key
        key = self._compute_key(item)
        # Check if element is already there.
        index = self._index.get(item)
        if index is None:
            # It is not: normal add
            self._heap.append((key, item))
            index = len(self._heap) - 1
            self._index[item] = index
            self._swim_up(index)
            is_new = True
        else:
            # Element is there
            oldkey, item = self._heap[index]
            if oldkey == key:
                # Keys are the same, nothing to do.
                pass
            else:
                # Replace value
                self._heap[index] = (key, item)
                # Compare value with parent and swim up or sink down accordingly.
                parent = self._parent(index)
                if parent is not None and key < self._heap[parent][0]:
                    self._swim_up(index)
                else:
                    self._sink_down(index)
            is_new = False
        return is_new

    def pop(self):
        """Removes and returns the element with the smallest key.

        :return: item with the smallest key
        """
        return self.pop_with_key()[1]

    def pop_with_key(self):
        """Removes and returns the smallest element and its key.

        :return: smallest element (key, element)
        """
        assert bool(self)
        # Get top element
        key, item = self._heap[0]
        # Swap top and last
        self._swap(0, len(self._heap) - 1)
        # Remove last element (former top)
        del self._index[item]
        self._heap.pop(-1)
        # Sink down new top element
        if self:
            self._sink_down(0)
        return key, item

    def peek(self):
        """Returns the element with the smallest key without removing it.

        :return: item with the smallest key
        """
        assert bool(self)
        # Get top element
        key, item = self._heap[0]
        return item

    def _parent(self, index):
        if index == 0:
            return None
        else:
            return (index - 1) // 2

    def _children(self, index):
        return (2 * index) + 1, (2 * index) + 2

    def _swim_up(self, index):
        p = self._parent(index)
        if p is not None and self._heap[p][0] > self._heap[index][0]:
            self._swap(p, index)
            self._swim_up(p)

    def _sink_down(self, index):
        c1, c2 = self._children(index)
        k1 = None
        k2 = None
        if c1 < len(self._heap):
            k1 = self._heap[c1][0]
            if c2 < len(self._heap):
                k2 = self._heap[c2][0]
        k = self._heap[index][0]
        if k1 is not None and k > k1:
            if k2 is not None and k1 > k2:
                self._swap(index, c2)
                self._sink_down(c2)
            else:
                self._swap(index, c1)
                self._sink_down(c1)
        elif k2 is not None and k > k2:
            self._swap(index, c2)
            self._sink_down(c2)

    def _swap(self, index1, index2):
        key1, item1 = self._heap[index1]
        key2, item2 = self._heap[index2]
        # Update index
        self._index[item1] = index2
        self._index[item2] = index1
        # Update heap
        self._heap[index1] = key2, item2
        self._heap[index2] = key1, item1

    def __len__(self):
        return len(self._heap)


class BitVector(object):

    def __init__(self):
        self.binsize_bits = 5
        self.binsize = 1 << self.binsize_bits
        # self.blocks = array.array('L')
        self.blocks = []
        self.blocks_size = []

    def add(self, index):
        # b = index // self.binsize
        # i = index % self.binsize
        mask = ((1 << self.binsize_bits) - 1)
        b = index >> self.binsize_bits
        i = index & mask
        n = len(self.blocks)
        if n <= b:
            self.blocks.extend([0] * (b - n + 1))
        self.blocks[b] |= (1 << i)

    def __contains__(self, index):
        mask = ((1 << self.binsize_bits) - 1)
        b = index >> self.binsize_bits
        i = index & mask
        n = len(self.blocks)
        if n <= b:
            return False
        else:
            return self.blocks[b] & (1 << i)

    def __iter__(self):
        o = 0
        for b, block in enumerate(self.blocks):
            if block != 0:
                for i in range(0, self.binsize):
                    if (1 << i) & block:
                        yield o + i
            o += self.binsize

    def __and__(self, other):
        result = BitVector()
        for a, b in zip(self.blocks, other.blocks):
            result.blocks.append(a & b)
        return result

    def __iand__(self, other):
        la = len(self.blocks)
        for i, b in enumerate(other.blocks[:la]):
            self.blocks[i] &= b
        return self

    def __or__(self, other):
        result = BitVector()
        for a, b in zip(self.blocks, other.blocks):
            result.blocks.append(a | b)

        la = len(self.blocks)
        lb = len(other.blocks)

        result.blocks.extend(self.blocks[lb:])
        result.blocks.extend(other.blocks[la:])
        return result

    def __ior__(self, other):
        la = len(self.blocks)
        for i, b in enumerate(other.blocks[:la]):
            self.blocks[i] |= b
        self.blocks.extend(other.blocks[la:])
        return self

    def __len__(self):
        n = 0
        for block in self.blocks:
            if block != 0:
                n += bin(block).count("1")
                # n += self._countbits(block)
        return n

    def __nonzero__(self):
        for b in self.blocks:
            if b:
                return True
        return False

    def __bool__(self):
        for b in self.blocks:
            if b:
                return True
        return False

    def _countbits(self, n):
        n = (n & 0x5555555555555555) + ((n & 0xAAAAAAAAAAAAAAAA) >> 1)
        n = (n & 0x3333333333333333) + ((n & 0xCCCCCCCCCCCCCCCC) >> 2)
        n = (n & 0x0F0F0F0F0F0F0F0F) + ((n & 0xF0F0F0F0F0F0F0F0) >> 4)
        n = (n & 0x00FF00FF00FF00FF) + ((n & 0xFF00FF00FF00FF00) >> 8)
        n = (n & 0x0000FFFF0000FFFF) + ((n & 0xFFFF0000FFFF0000) >> 16)
        n = (n & 0x00000000FFFFFFFF) + ((n & 0xFFFFFFFF00000000) >> 32)  # This last & isn't strictly necessary.
        return n