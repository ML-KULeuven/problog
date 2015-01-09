"""Simple HTTP server for ProbLog.

This module only defines the server interface.
To change settings related to ProbLog, you should consult the run_problog.py script.

The operation of the server is determined by four options:

    port (default: 8000)        Server port to use
    timeout (default: 60)       Maximum *processing* time of the ProbLog subprocess
    memout (default: 1Gb)       Maximum memory usage of the ProbLog subprocess
    servefiles (default: No)    Whether to serve a file for undefined paths. (This is potentially unsafe.)
    
The server defines one path:

    http://hostname:port/problog?model=...
    
This path can be used with GET or POST requests.

If ``servefiles`` is enabled other paths will be treated as file access requests.

Copyright 2015 Anton Dries, Wannes Meert.
All rights reserved.

"""

from __future__ import print_function

# Load general Python modules
import sys
import os
import glob
import tempfile
import traceback
import subprocess
import resource
import logging, logging.config
import json

DEFAULT_PORT = 5100
DEFAULT_TIMEOUT = 60
DEFAULT_MEMOUT = 1.0 # gigabyte

SERVE_FILES=False
CACHE_MODELS=True
CACHE_DIR="cache"

api_root = '/'

here = os.path.dirname(__file__)
logging.config.fileConfig(os.path.join(here,'logging.conf'))
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
else :
    import http.server as BaseHTTPServer    
    import urllib.parse as urlparse
    def toBytes( string ) :
        return bytes(string, 'UTF-8')
    url_decode = urlparse.unquote_plus

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
      import hashlib
      inhash = hashlib.md5(model.encode()).hexdigest()
      if not os.path.exists(CACHE_DIR):
          os.mkdir(CACHE_DIR)
      infile = os.path.join(CACHE_DIR, inhash+'.pl')
      outfile = os.path.join(CACHE_DIR, inhash+'.out')
    else:
      handle, infile = tempfile.mkstemp('.pl')
      handle, outfile = tempfile.mkstemp('.out')

    with open(infile, 'w') as f :
        f.write(model)


    cmd = [ 'python', root_path('run_problog.py'), infile, outfile ]

    try :
        call_process(cmd, DEFAULT_TIMEOUT, DEFAULT_MEMOUT * (1 << 30))

        with open(outfile) as f :
            result = f.read()
        code, datatype, datavalue = result.split(None,2)

        print("RETURN:")
        print(code, datatype, datavalue)

        if datatype == 'application/json':
            datavalue = '{}({});'.format(callback, datavalue)

        return int(code), datatype, datavalue
    except subprocess.CalledProcessError :
        return 500, 'application/json', json.dumps('ProbLog evaluation exceeded time or memory limit')

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

    if False and CACHE_MODELS:  # Disabled for now
      import hashlib
      inhash = hashlib.md5(model).hexdigest()
      if not os.path.exists(CACHE_DIR):
          os.mkdir(CACHE_DIR)
      infile = os.path.join(CACHE_DIR, inhash+'.pl')
      outfile = os.path.join(CACHE_DIR, inhash+'.out')
    else:
      handle, infile = tempfile.mkstemp('.pl')
      handle, datafile = tempfile.mkstemp('.data')
      handle, outfile = tempfile.mkstemp('.out')

    with open(infile, 'w') as f :
        f.write(model)

    with open(datafile, 'w') as f :
        f.write(examples)

    cmd = [ 'python', root_path('run_learning.py'), infile, datafile, outfile ]

    try :
        call_process(cmd, DEFAULT_TIMEOUT, DEFAULT_MEMOUT * (1 << 30))

        with open(outfile) as f :
            result = f.read()
        code, datatype, datavalue = result.split(None,2)

        if datatype == 'application/json':
            datavalue = '{}({});'.format(callback, datavalue)

        return int(code), datatype, datavalue
    except subprocess.CalledProcessError :
        import json
        return 500, 'application/json', json.dumps('ProbLog evaluation exceeded time or memory limit')
    


@handle_url(api_root+'model')
def get_model_from_hash_jsonp(hash, callback):
    hash = hash[0]
    callback = callback[0]
    infile = os.path.join(CACHE_DIR, hash+'.pl')

    if not CACHE_MODELS or not os.path.exists(infile):
        return 500, 'text/plain', 'Model hash not available'

    result = dict()
    with open(infile, 'r') as f:
         result['model'] = f.read()

    import json
    datatype = 'application/json'
    datavalue = json.dumps(result)
    datavalue = '{}({});'.format(callback, datavalue)
    code = 200

    return int(code), datatype, datavalue


class ProbLogHTTP(BaseHTTPServer.BaseHTTPRequestHandler) :

    def do_POST(self) :
        numberOfBytes = int(self.headers["Content-Length"])
        data = self.rfile.read(numberOfBytes)
        query = urlparse.parse_qs(data)

        logger.info('POST - {}'.format(query))

        self.do_GET(query=query)


    def do_GET(self, query=None) :
        """Handle a GET request.

        Looks up the URL path in PATHS and executes the corresponding function.
        If the path does not occur in PATHS, treats the path as a filename and serves the file.
        """

        url = urlparse.urlparse( self.path )
        path = url.path
        if query == None :
            query = urlparse.parse_qs(url.query)
        if '_' in query:
            # Used by jquery to avoid caching
            del query['_']

        logger.info('GET - {} - {}'.format(path, self.client_address[0]))

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


if __name__ == '__main__' :
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, default=DEFAULT_PORT, help="Server listening port")
    parser.add_argument('--timeout', '-t', type=int, default=DEFAULT_TIMEOUT, help="Time limit in seconds")
    parser.add_argument('--memout', '-m', type=float, default=DEFAULT_MEMOUT, help="Memory limit in Gb")
    parser.add_argument('--servefiles', '-F', action='store_true', help="Attempt to serve a file for undefined paths (unsafe?).")
    parser.add_argument('--nocaching', action='store_true', help="Disable caching of submitted models")
    args = parser.parse_args(sys.argv[1:])

    DEFAULT_TIMEOUT = args.timeout
    DEFAULT_MEMOUT = args.memout
    SERVE_FILES = args.servefiles
    CACHE_MODELS = not args.nocaching
    print('Starting server on port %d (timeout=%d, memout=%dGb)' % (args.port, DEFAULT_TIMEOUT, DEFAULT_MEMOUT ))    
    logger.info('Starting server on port %d (timeout=%d, memout=%dGb)' % (args.port, DEFAULT_TIMEOUT, DEFAULT_MEMOUT ))    

    server_address = ('', args.port)
    httpd = BaseHTTPServer.HTTPServer( server_address, ProbLogHTTP )
    httpd.serve_forever()

