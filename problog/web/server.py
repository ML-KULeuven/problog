#!/usr/bin/env python
# encoding: utf-8
"""Simple HTTP server for ProbLog.

This module only defines the server interface.
To change settings related to ProbLog, you should consult the run_problog.py script.

The operation of the server is determined by four options:

    port (default: 8000)        Server port to use
    timeout (default: 60)       Maximum *processing* time of the ProbLog subprocess
    memout (default: 1Gb)       Maximum memory usage of the ProbLog subprocess
    servefiles (default: No)    Whether to serve a file for undefined paths. (This is potentially unsafe.)

The server defines the following paths:

    http://hostname:port/problog?model=... [GET,POST]
    http://hostname:port/inference?model=...&callback=... [JSONP]
    http://hostname:port/learn?model=...&examples=...&callback=... [JSONP]

If ``servefiles`` is enabled other paths will be treated as file access requests.

__author__ = Anton Dries
__author__ = Wannes Meert

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

# Load general Python modules
import sys
import os
import tempfile
import traceback
import subprocess
import resource
import logging
import logging.config

DEFAULT_PORT = 5100
DEFAULT_TIMEOUT = 60
DEFAULT_MEMOUT = 1.0  # gigabyte

SERVE_FILES = False
CACHE_MODELS = True
CACHE_DIR = "cache"

PYTHON_EXEC = 'python'    # Python 2
# PYTHON_EXEC = sys.executable  # Match with server

api_root = '/'

here = os.path.dirname(__file__)

try :
    logging.config.fileConfig(os.path.join(here,'logging.conf'))
except IOError :
    pass
logger = logging.getLogger('server')

# Load Python standard web-related modules (based on Python version)
import json
import mimetypes
if sys.version_info.major == 2 :
    import BaseHTTPServer
    import urlparse
    def toBytes( string ) :
        return bytes(string)
    import urllib
    url_decode = urllib.unquote_plus
    import hashlib
    def compute_hash(model) :
        return hashlib.md5(model).hexdigest() # Python 2

else :
    import http.server as BaseHTTPServer
    import urllib.parse as urlparse
    def toBytes( string ) :
        return bytes(string, 'UTF-8')
    url_decode = urlparse.unquote_plus
    import hashlib
    def compute_hash(model) :
        return hashlib.md5(toBytes(model)).hexdigest() # Python 3

# Contains special URL paths. Initialize with @handle_url
PATHS = {}

class handle_url(object) :
    """Decorator for adding a handler for a URL.

    Example: to add a handler for GET /hello?name=World

    @handle_url('/hello')
    def hello(name) :
        return 200, 'text/plain', 'Hello %s!' % name
    """

    def __init__(self, path) :
        self.path = path

    def __call__(self, f) :
        PATHS[self.path] = f


def root_path(relative_path) :
    """Translate URL path to file system path."""
    return os.path.abspath( os.path.join(os.path.dirname(__file__), relative_path.lstrip('/') ) )

def model_path(*args) :
    """Translate model path to file system path."""
    return os.path.abspath( os.path.join(os.path.dirname(__file__), '../test/', *args ) )

def call_process( cmd, timeout, memout ) :
    """Call a subprocess with time and memory restrictions.

        Note:
            - the time restriction only works on Linux and Mac OS X
            - the memory restriction only works on Linux

        :param:timeout: CPU time in seconds
        :param:memout: Maximum memory in bytes
        :return: Output of the subprocess.

    """

    def setlimits() :
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
        resource.setrlimit(resource.RLIMIT_AS, (memout, memout))

    return subprocess.check_output(cmd, preexec_fn=setlimits)

def wrap_callback(callback, jsonstr):
    return '{}({});'.format(callback, jsonstr)

#@handle_url('/problog')
#def run_problog( model ) :
    #"""Evaluate the given model and return the probabilities."""
    #model = model[0]

    #handle, tmpfile = tempfile.mkstemp('.pl')
    #with open(tmpfile, 'w') as f :
        #f.write(model)

    #handle, outfile = tempfile.mkstemp('.out')

    #cmd = [ 'python', root_path('run_problog.py'), tmpfile, outfile ]

    #try :
        #call_process(cmd, DEFAULT_TIMEOUT, DEFAULT_MEMOUT * (1 << 30))

        #with open(outfile) as f :
            #result = f.read()
        #code, datatype, datavalue = result.split(None,2)
        #return int(code), datatype, datavalue
    #except subprocess.CalledProcessError :
        #return 500, 'text/plain', 'ProbLog evaluation exceeded time or memory limit'


@handle_url(api_root+'inference')
def run_problog_jsonp(model, callback):
    """Evaluate the given model and return the probabilities using the JSONP
       standard.
       This mode is required to send an API request when Problog is
       running on a different server (to avoid cross-side-scripting
       limitations).
    """
    model = model[0]
    callback = callback[0]

    if CACHE_MODELS:
      try:
          inhash = compute_hash(model)
      except UnicodeDecodeError as e:
          logger.error('Unicode error catched: {}'.format(e))
          return 200, 'application/json', wrap_callback(callback, json.dumps({'SUCCESS':False,'err':'Cannot decode character in program: {}'.format(e)}))
      if not os.path.exists(CACHE_DIR):
          os.mkdir(CACHE_DIR)
      infile = os.path.join(CACHE_DIR, inhash+'.pl')
      outfile = os.path.join(CACHE_DIR, inhash+'.out')
      logger.info('Saved file: {}'.format(infile))
    else:
      _, infile = tempfile.mkstemp('.pl')
      _, outfile = tempfile.mkstemp('.out')

    with open(infile, 'w') as f :
        f.write(model)


    cmd = [ PYTHON_EXEC, root_path('run_problog.py'), infile, outfile ]

    try :
        call_process(cmd, DEFAULT_TIMEOUT, DEFAULT_MEMOUT * (1 << 30))

        with open(outfile) as f :
            result = f.read()
        code, datatype, datavalue = result.split(None,2)

        if datatype == 'application/json':
            datavalue = '{}({});'.format(callback, datavalue)

        return int(code), datatype, datavalue
    except subprocess.CalledProcessError as err:
        result = {'SUCCESS': False, 'err': 'ProbLog evaluation exceeded time or memory limit (code: %s)' % err.returncode}
        return 200, 'application/json', wrap_callback(callback, json.dumps(result))


@handle_url(api_root + 'mpe')
def run_mpe_jsonp(model, callback):
    """Evaluate the given model and return the probabilities using the JSONP
       standard.
       This mode is required to send an API request when Problog is
       running on a different server (to avoid cross-side-scripting
       limitations).
    """
    model = model[0]
    callback = callback[0]

    if CACHE_MODELS:
        try:
            inhash = compute_hash(model)
        except UnicodeDecodeError as e:
            logger.error('Unicode error catched: {}'.format(e))
            result = {'SUCCESS': False, 'err': 'Cannot decode character in program: {}'.format(e)}
            return 200, 'application/json', wrap_callback(callback, json.dumps(result))
        if not os.path.exists(CACHE_DIR):
            os.mkdir(CACHE_DIR)
        infile = os.path.join(CACHE_DIR, inhash + '.pl')
        outfile = os.path.join(CACHE_DIR, inhash + '.out')
        logger.info('Saved file: {}'.format(infile))
    else:
        _, infile = tempfile.mkstemp('.pl')
        _, outfile = tempfile.mkstemp('.out')

    with open(infile, 'w') as f:
        f.write(model)

    cmd = [PYTHON_EXEC, root_path('run_problog.py'), 'mpe', '--full', infile, outfile]

    try:
        call_process(cmd, DEFAULT_TIMEOUT, DEFAULT_MEMOUT * (1 << 30))

        with open(outfile) as f:
            result = f.read()
        code, datatype, datavalue = result.split(None, 2)

        if datatype == 'application/json':
            datavalue = '{}({});'.format(callback, datavalue)

        return int(code), datatype, datavalue
    except subprocess.CalledProcessError as err:
        result = {'SUCCESS': False, 'err': 'ProbLog evaluation exceeded time or memory limit (code: %s)' % err.returncode}
        return 200, 'application/json', wrap_callback(callback, json.dumps(result))


@handle_url(api_root + 'sample')
def run_mpe_jsonp(model, callback):
    """Evaluate the given model and return the probabilities using the JSONP
       standard.
       This mode is required to send an API request when Problog is
       running on a different server (to avoid cross-side-scripting
       limitations).
    """
    model = model[0]
    callback = callback[0]

    if CACHE_MODELS:
        try:
            inhash = compute_hash(model)
        except UnicodeDecodeError as e:
            logger.error('Unicode error catched: {}'.format(e))
            result = {'SUCCESS': False, 'err': 'Cannot decode character in program: {}'.format(e)}
            return 200, 'application/json', wrap_callback(callback, json.dumps(result))
        if not os.path.exists(CACHE_DIR):
            os.mkdir(CACHE_DIR)
        infile = os.path.join(CACHE_DIR, inhash + '.pl')
        outfile = os.path.join(CACHE_DIR, inhash + '.out')
        logger.info('Saved file: {}'.format(infile))
    else:
        _, infile = tempfile.mkstemp('.pl')
        _, outfile = tempfile.mkstemp('.out')

    with open(infile, 'w') as f:
        f.write(model)

    cmd = [PYTHON_EXEC, root_path('run_problog.py'), 'sample', infile, outfile]

    try:
        call_process(cmd, DEFAULT_TIMEOUT, DEFAULT_MEMOUT * (1 << 30))

        with open(outfile) as f:
            result = f.read()
        code, datatype, datavalue = result.split(None, 2)

        if datatype == 'application/json':
            datavalue = '{}({});'.format(callback, datavalue)

        return int(code), datatype, datavalue
    except subprocess.CalledProcessError as err:
        result = {'SUCCESS': False, 'err': 'ProbLog evaluation exceeded time or memory limit (code: %s)' % err.returncode}
        return 200, 'application/json', wrap_callback(callback, json.dumps(result))


@handle_url(api_root+'learning')
def run_learning_jsonp(model, examples, callback) :
    """Evaluate the given model and return the probabilities using the JSONP
       standard.
       This mode is required to send an API request when Problog is
       running on a different server (to avoid cross-side-scripting
       limitations).
    """
    model = model[0]
    examples = examples[0]
    callback = callback[0]

    if CACHE_MODELS:
      import hashlib
      try:
          inhash = hashlib.md5(model.decode('utf-8')).hexdigest()
      except UnicodeDecodeError as e:
          logger.error('Unicode error catched: {}'.format(e))
          return 200, 'application/json', wrap_callback(callback, json.dumps({'SUCCESS':False,'err':'Cannot decode character in program: {}'.format(e)}))
      try:
          datahash = hashlib.md5(examples.decode('utf-8')).hexdigest()
      except UnicodeDecodeError as e:
          logger.error('Unicode error catched: {}'.format(e))
          return 200, 'application/json', wrap_callback(callback, json.dumps({'SUCCESS':False,'err':'Cannot decode character in examples: {}'.format(e)}))
      if not os.path.exists(CACHE_DIR):
          os.mkdir(CACHE_DIR)
      infile = os.path.join(CACHE_DIR, inhash+'.pl')
      datafile = os.path.join(CACHE_DIR, datahash+'.data')
      logger.info('Saved files: {} and {}'.format(infile, datafile))
      outfile = os.path.join(CACHE_DIR, inhash+'_'+datahash+'.out')
    else:
      _, infile = tempfile.mkstemp('.pl')
      _, datafile = tempfile.mkstemp('.data')
      _, outfile = tempfile.mkstemp('.out')

    with open(infile, 'w') as f :
        f.write(model)

    with open(datafile, 'w') as f :
        f.write(examples)

    cmd = [ PYTHON_EXEC, root_path('run_learning.py'), infile, datafile, outfile ]

    try :
        call_process(cmd, DEFAULT_TIMEOUT, DEFAULT_MEMOUT * (1 << 30))

        with open(outfile) as f :
            result = f.read()
        code, datatype, datavalue = result.split(None,2)

        if datatype == 'application/json':
            datavalue = wrap_callback(callback, datavalue)

        return int(code), datatype, datavalue
    except subprocess.CalledProcessError :
        return 200, 'application/json', wrap_callback(callback, json.dumps({'SUCCESS':False, 'err':'ProbLog learning exceeded time or memory limit'}))


@handle_url(api_root+'model')
def get_model_from_hash_jsonp(hash, callback):
    hash = hash[0]
    callback = callback[0]
    infile = os.path.join(CACHE_DIR, hash+'.pl')

    if not CACHE_MODELS or not os.path.exists(infile):
        return 200, 'application/json', wrap_callback(callback, json.dumps({'SUCCESS': False, 'err': 'Model hash not available: {}'.format(hash)}))

    result = {'SUCCESS': True}
    with open(infile, 'r') as f:
         result['model'] = f.read()

    datatype = 'application/json'
    datavalue = json.dumps(result)
    datavalue = wrap_callback(callback, datavalue)
    code = 200

    return int(code), datatype, datavalue


@handle_url(api_root+'examples')
def get_example_from_hash_jsonp(ehash, callback):
    ehash = ehash[0]
    callback = callback[0]
    infile = os.path.join(CACHE_DIR, ehash+'.data')

    if not CACHE_MODELS or not os.path.exists(infile):
        return 200, 'application/json', wrap_callback(callback, json.dumps({'SUCCESS': False, 'err': 'Examples hash not available: {}'.format(ehash)}))

    result = {'SUCCESS': True}
    with open(infile, 'r') as f:
         result['examples'] = f.read()

    datatype = 'application/json'
    datavalue = json.dumps(result)
    datavalue = wrap_callback(callback, datavalue)
    code = 200

    return int(code), datatype, datavalue


class ProbLogHTTP(BaseHTTPServer.BaseHTTPRequestHandler) :

    def do_POST(self) :
        numberOfBytes = int(self.headers["Content-Length"])
        data = self.rfile.read(numberOfBytes)
        query = urlparse.parse_qs(data)

        self.do_GET(query=query)


    def do_GET(self, query=None) :
        """Handle a GET request.

        Looks up the URL path in PATHS and executes the corresponding function.
        If the path does not occur in PATHS, treats the path as a filename and serves the file.
        """

        try:
            url = urlparse.urlparse( self.path )
            path = url.path
            if query == None :
                query = urlparse.parse_qs(url.query)
            if '_' in query:
                # Used by jquery to avoid caching
                del query['_']

            action = PATHS.get(path)
            if action == None :
                if SERVE_FILES :
                    self.serveFile(path)
                else :
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(toBytes('File not found!'))
            else :
                code, datatype, data = action( **query )
                self.send_response(code)
                self.send_header("Content-type", datatype)
                self.end_headers()
                if data :
                    self.wfile.write(toBytes(data))
        except Exception as e:
          import traceback
          logger.error('Uncaught exception: {}\n{}'.format(e, traceback.format_exc()))

    def serveFile(self, filename) :
        """Serve a file."""
        if filename == '/' : filename = '/index.html'

        filetype, encoding = mimetypes.guess_type(filename)
        filename = root_path(filename)
        try :
            with open(filename) as f :
                data = f.read()
            self.send_response(200)
            self.send_header("Content-type", filetype)
            if encoding :
                self.send_header("Content-Encoding", encoding)
            self.end_headers()
            self.wfile.write(toBytes(data))
        except :
            self.send_response(404)
            self.end_headers()
            self.wfile.write(toBytes('File not found!'))

    def log_message(self,format, *args) :
        try :
            # Remove arguments from GET request, only keep path
            if args[0].startswith('GET') :
                p = args[0].find('?')
                if p >= 0 :
                    args0 = args[0][0:p]
                else :
                    args0 = args[0]
                args = (args0,) + args[1:]
        except Exception :
            pass

        args = (self.client_address[0],) + args
        format = '[%s] ' + format
        logger.info(format % args)


def main(argv, **extra):
    global DEFAULT_MEMOUT
    global DEFAULT_TIMEOUT
    global SERVE_FILES
    global CACHE_MODELS

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, default=DEFAULT_PORT, help="Server listening port")
    parser.add_argument('--timeout', '-t', type=int, default=DEFAULT_TIMEOUT, help="Time limit in seconds")
    parser.add_argument('--memout', '-m', type=float, default=DEFAULT_MEMOUT, help="Memory limit in Gb")
    parser.add_argument('--servefiles', '-F', action='store_true', help="Attempt to serve a file for undefined paths (unsafe?).")
    parser.add_argument('--nocaching', action='store_true', help="Disable caching of submitted models")
    parser.add_argument('--browser', action='store_true', help="Open editor in web browser.")
    args = parser.parse_args(argv)

    DEFAULT_TIMEOUT = args.timeout
    DEFAULT_MEMOUT = args.memout
    SERVE_FILES = args.servefiles
    CACHE_MODELS = not args.nocaching
    logger.info('Starting server on port %d (timeout=%d, memout=%dGb)' % (args.port, DEFAULT_TIMEOUT, DEFAULT_MEMOUT))

    server_address = ('', args.port)
    httpd = BaseHTTPServer.HTTPServer(server_address, ProbLogHTTP)
    if args.browser :
        import webbrowser
        webbrowser.open( 'file://' + os.path.join(os.path.abspath(here), 'index_local.html'), new=2, autoraise=True)
    httpd.serve_forever()


if __name__ == '__main__':
    main(sys.argv[1:])
