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
DEFAULT_MEMOUT = 2.0  # gigabyte

SERVE_FILES = False
CACHE_MODELS = True
CACHE_DIR = "cache"

RUN_LOCAL = False

# PYTHON_EXEC = 'python'    # Python 2
PYTHON_EXEC = sys.executable  # Match with server

api_root = '/'

FILES_WHITELIST = [
    api_root + 'js/problog_editor_advanced.js',
    api_root + 'js/lib/ace.js',
    api_root + 'js/lib/bootstrap-theme.min.css',
    api_root + 'js/lib/bootstrap.min.css',
    api_root + 'js/lib/bootstrap.min.js',
    api_root + 'js/lib/jquery-2.1.0.min.js',
    api_root + 'js/lib/jquery-ui.min.js',
    api_root + 'js/lib/mode-prolog.js',
    api_root + 'tutorial.html',
    api_root + 'sidebar.css'
]

here = os.path.dirname(__file__)

try :
    logging.config.fileConfig(os.path.join(here,'logging.conf'))
except IOError:
    logger = logging.getLogger('server')
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
logger = logging.getLogger('server')

# Load Python standard web-related modules (based on Python version)
import json
import mimetypes
if sys.version_info.major == 2:
    import BaseHTTPServer
    import urlparse

    def to_bytes(string):
        return bytes(string)
    import urllib
    url_decode = urllib.unquote_plus
    import hashlib

    def compute_hash(model):
        return hashlib.md5(model).hexdigest()  # Python 2

else:
    import http.server as BaseHTTPServer
    import urllib.parse as urlparse

    def to_bytes(string):
        return bytes(string, 'UTF-8')
    url_decode = urlparse.unquote_plus
    import hashlib

    def compute_hash(model):
        return hashlib.md5(to_bytes(model)).hexdigest() # Python 3

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


def root_path(*relative_path):
    """Translate URL path to file system path."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *relative_path))

PROB_EXEC = root_path('..', 'tasks', '__init__.py')


def call_process(cmd, timeout, memout):
    """Call a subprocess with time and memory restrictions.

        Note:
            - the time restriction only works on Linux and Mac OS X
            - the memory restriction only works on Linux

        :param:timeout: CPU time in seconds
        :param:memout: Maximum memory in bytes
        :return: Output of the subprocess.

    """

    def setlimits():
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
        resource.setrlimit(resource.RLIMIT_AS, (memout, memout))

    return subprocess.check_output(cmd, preexec_fn=setlimits)


def wrap_callback(callback, jsonstr):
    return '{}({});'.format(callback, jsonstr)


def run_problog_task(task, model, callback=None, data=None, options=None):
    if options is None:
        options = []

    try:
        # Write the model to a temporary or cached file.
        infile = store_hash(model, 'pl')

        # Construct the basic command
        cmd = [PYTHON_EXEC, PROB_EXEC, task, infile]

        # Write the data to a temporary or cached file (if given)
        datafile = None
        if data is not None:
            datafile = store_hash(data, 'data')
            outfile = os.path.splitext(infile)[0] + '_' + \
                os.path.splitext(os.path.basename(datafile))[0] + '.out'
            cmd += [datafile]
        else:
            outfile = os.path.splitext(infile)[0] + '.out'
        logger.info('Output file: {}'.format(outfile))

        # Add default and given options to the command.
        cmd += ['-o', outfile, '--web'] + options

    except UnicodeDecodeError as err:
        logger.error('Unicode error catched: {}'.format(err))
        result = {'SUCCESS': False, 'err': 'Cannot decode character in program: {}'.format(err)}
        return 200, 'application/json', wrap_callback(callback, json.dumps(result))

    if CACHE_MODELS:
        url = 'task=%s' % task
        url += '&hash=%s' % os.path.splitext(os.path.basename(infile))[0]
        if data is not None:
            url += '&ehash=%s' % os.path.splitext(os.path.basename(datafile))[0]
    else:
        url = None

    try:
        # Execute ProbLog
        call_process(cmd, DEFAULT_TIMEOUT, DEFAULT_MEMOUT * (1 << 30))

        # Read output produced by ProbLog
        with open(outfile) as f:
            result = f.read()

        if url is not None:
            result = json.loads(result)
            result['url'] = url
            result = json.dumps(result)

        # Wrap the output in a JSON wrapper.
        datavalue = wrap_callback(callback, result)
        return 200, 'application/json', datavalue
    except subprocess.CalledProcessError as err:
        logger.error('ProbLog didn\'t finish correctly: %s' % err)
        result = {'SUCCESS': False, 'url': url,
                  'err': 'ProbLog exceeded time or memory limit'}
        return 200, 'application/json', wrap_callback(callback, json.dumps(result))


@handle_url(api_root + 'inference')
def run_problog_jsonp(model, callback):
    return run_problog_task('prob', model[0], callback[0])


@handle_url(api_root + 'prob')
def run_problog_jsonp(model, callback):
    return run_problog_task('prob', model[0], callback[0])


@handle_url(api_root + 'dt')
def run_problog_jsonp(model, callback, solve=None):
    if solve is not None and solve[0] == 'local':
        options = ['--search', 'local']
    else:
        options = []
    return run_problog_task('dt', model[0], callback[0], options=options)


@handle_url(api_root + 'mpe')
def run_mpe_jsonp(model, callback):
    return run_problog_task('mpe', model[0], callback[0], options=['--full'])


@handle_url(api_root + 'sample')
def run_sample_jsonp(model, callback):
    return run_problog_task('sample', model[0], callback[0])


@handle_url(api_root + 'explain')
def run_sample_jsonp(model, callback):
    return run_problog_task('explain', model[0], callback[0])


@handle_url(api_root + 'lfi')
def run_lfi_jsonp(model, examples, callback):
    return run_problog_task('lfi', model[0], callback[0], data=examples[0])


@handle_url(api_root + 'learning')
def run_lfi_jsonp(model, examples, callback):
    return run_problog_task('lfi', model[0], callback[0], data=examples[0])


@handle_url(api_root + 'ground')
def run_lfi_jsonp(model, callback):
    return run_problog_task('ground', model[0], callback[0], options=['--format', 'svg'])


def store_hash(data, datatype):
    # Can raise UnicodeDecodeError
    if CACHE_MODELS:
        datahash = compute_hash(data)
        if not os.path.exists(CACHE_DIR):
            os.mkdir(CACHE_DIR)
        datafile = os.path.join(CACHE_DIR, datahash + '.' + datatype)
        logger.info('Saved file: {}'.format(datafile))
    else:
        _, datafile = tempfile.mkstemp('.' + datatype)

    with open(datafile, 'w') as f:
        f.write(data)

    return datafile


def get_hash(hashcode, datatype):
    datafile = os.path.join(CACHE_DIR, hashcode + '.' + datatype)
    if not CACHE_MODELS or not os.path.exists(datafile):
        return None
    else:
        with open(datafile, 'r') as f:
            data = f.read()
        return data


@handle_url(api_root + 'model')
def get_model_from_hash_jsonp(hash, callback):
    data = get_hash(hash[0], 'pl')
    if data is None:
        result = {'SUCCESS': False, 'err': 'Model hash not available: {}'.format(hash[0])}
    else:
        result = {'SUCCESS': True, 'model': data}
    return 200, 'application/json', wrap_callback(callback[0], json.dumps(result))


@handle_url(api_root + 'examples')
def get_data_from_hash_jsonp(hash, callback):
    data = get_hash(hash[0], 'data')
    if data is None:
        result = {'SUCCESS': False, 'err': 'Model hash not available: {}'.format(hash[0])}
    else:
        result = {'SUCCESS': True, 'examples': data}
    return 200, 'application/json', wrap_callback(callback[0], json.dumps(result))


@handle_url(api_root + 'models')
def get_example_models(callback):
    return


@handle_url(api_root)
def get_editor():
    with open(root_path('editor_adv.html')) as f:
        data = f.read()
    data = data.replace('problog.init()', 'problog.init(\'\');')
    data = make_local(data)
    return 200, 'text/html', data


def make_local(html):
    import re

    if RUN_LOCAL:
        html = re.sub('<link rel="stylesheet" href="(http([^/"]*/)+)', '<link rel="stylesheet" href="js/lib/', html)
        html = re.sub('<script src="(http([^/"]*/)+)', '<script src="js/lib/', html)
        return html
    else:
        return html


# @handle_url(api_root + 'js/problog_editor_advanced.js')
# def get_javascript():
#     with open(root_path('js/problog_editor_advanced.js')) as f:
#         data = f.read()
#     return 200, 'application/javascript', data
#
#
# @handle_url(api_root + 'tutorial.html')
# def get_javascript():
#     with open(root_path('js/problog_editor_advanced.js')) as f:
#         data = f.read()
#     return 200, 'application/javascript', data

class ProbLogHTTP(BaseHTTPServer.BaseHTTPRequestHandler):

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
            if query is None:
                query = urlparse.parse_qs(url.query)
            if '_' in query:
                # Used by jquery to avoid caching
                del query['_']

            action = PATHS.get(path)
            if action is None:
                if SERVE_FILES or str(path) in FILES_WHITELIST:
                    self.serveFile(path)
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(to_bytes('File not found!'))
            else :
                code, datatype, data = action(**query)
                self.send_response(code)
                self.send_header("Content-type", datatype)
                self.end_headers()
                if data:
                    self.wfile.write(to_bytes(data))
        except Exception as e:
            import traceback
            logger.error('Uncaught exception: {}\n{}'.format(e, traceback.format_exc()))

    def serveFile(self, filename) :
        """Serve a file."""
        if filename == '/':
            filename = '/index.html'
        filename = filename.lstrip('/')

        filetype, encoding = mimetypes.guess_type(filename)
        filename = root_path(filename)
        try :
            with open(filename) as f:
                data = f.read()
            data = make_local(data)
            self.send_response(200)
            self.send_header("Content-type", filetype)
            if encoding :
                self.send_header("Content-Encoding", encoding)
            self.end_headers()
            self.wfile.write(to_bytes(data))
        except :
            self.send_response(404)
            self.end_headers()
            self.wfile.write(to_bytes('File not found!'))

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
    global RUN_LOCAL

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, default=DEFAULT_PORT, help="Server listening port")
    parser.add_argument('--timeout', '-t', type=int, default=DEFAULT_TIMEOUT, help="Time limit in seconds")
    parser.add_argument('--memout', '-m', type=float, default=DEFAULT_MEMOUT, help="Memory limit in Gb")
    parser.add_argument('--servefiles', '-F', action='store_true', help="Attempt to serve a file for undefined paths (unsafe?).")
    parser.add_argument('--nocaching', action='store_true', help="Disable caching of submitted models")
    parser.add_argument('--browser', '-B', action='store_true', help="Open editor in web browser.")
    parser.add_argument('--local', '-l', action='store_true', help="Use local javascript libraries.")
    args = parser.parse_args(argv)

    RUN_LOCAL = args.local
    if args.local:
        args.servefiles = True

    DEFAULT_TIMEOUT = args.timeout
    DEFAULT_MEMOUT = args.memout
    SERVE_FILES = args.servefiles
    CACHE_MODELS = not args.nocaching
    logger.info('Starting server on port %d (timeout=%d, memout=%dGb)' % (args.port, DEFAULT_TIMEOUT, DEFAULT_MEMOUT))

    server_address = ('', args.port)
    httpd = BaseHTTPServer.HTTPServer(server_address, ProbLogHTTP)
    if args.browser:
        import webbrowser
        webbrowser.open('http://localhost:%s/' % args.port, new=2, autoraise=True)
    httpd.serve_forever()


if __name__ == '__main__':
    main(sys.argv[1:])
